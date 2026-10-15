"""Image-derived crop-row spacing estimation.

The estimator reports a distribution summary instead of treating a single
distance as universal.  All search scales are inferred from the center map;
meters enter only through the raster resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from rowgraphnet.domain import SpacingEstimate


@dataclass(slots=True)
class SpacingField:
    estimates: list[SpacingEstimate]
    reliable: bool
    method: str = "transverse_autocorrelation"

    @property
    def global_estimate(self) -> SpacingEstimate | None:
        if not self.estimates:
            return None
        weights = np.array([max(item.confidence, 1e-12) for item in self.estimates])
        medians = np.array([item.median_m for item in self.estimates])
        order = np.argsort(medians)
        cumulative = np.cumsum(weights[order])
        selected = order[np.searchsorted(cumulative, cumulative[-1] / 2.0)]
        return self.estimates[int(selected)]


def _transverse_profile(
    center: np.ndarray,
    theta_rad: float,
    confidence: np.ndarray | None,
) -> np.ndarray:
    rows, cols = np.indices(center.shape, dtype=np.float64)
    # A line tangent is (cos theta, sin theta) in image (x, y).  Dotting with
    # its normal produces a coordinate across the rows.
    transverse = -cols * np.sin(theta_rad) + rows * np.cos(theta_rad)
    transverse -= transverse.min()
    bins = np.rint(transverse).astype(np.int64)
    weights = np.asarray(center, dtype=np.float64)
    if confidence is not None:
        weights = weights * np.clip(confidence, 0.0, 1.0)
    sums = np.bincount(bins.ravel(), weights=weights.ravel())
    counts = np.bincount(bins.ravel(), weights=(weights > 0).ravel())
    return sums / np.maximum(counts, 1.0)


def _robust_entropy(scores: np.ndarray) -> float:
    positive = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    total = positive.sum()
    if total <= np.finfo(float).eps or len(positive) <= 1:
        return 1.0
    probabilities = positive / total
    value = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    return float(value / np.log(len(probabilities)))


def _estimate_profile_spacing(profile: np.ndarray, pixel_size_m: float) -> SpacingEstimate | None:
    finite = np.isfinite(profile)
    if finite.sum() < 3 or np.nanmax(profile) <= np.nanmin(profile):
        return None
    clean = np.where(finite, profile, np.nanmedian(profile[finite]))
    centered = clean - np.median(clean)
    spread = np.subtract(*np.percentile(centered, [75, 25]))
    if spread <= np.finfo(float).eps:
        return None

    # Smoothing is derived from the profile's own run lengths above its median.
    active = clean > np.median(clean)
    transitions = np.flatnonzero(np.diff(np.r_[False, active, False]))
    run_lengths = np.diff(transitions)[::2] if len(transitions) >= 2 else np.array([1])
    row_width_px = max(float(np.median(run_lengths)), 1.0)
    smoothed = gaussian_filter1d(centered, sigma=max(row_width_px / 4.0, 0.5), mode="nearest")

    spectrum = np.fft.rfft(smoothed, n=2 * len(smoothed))
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum))[: len(smoothed)]
    overlap = np.arange(len(smoothed), 0, -1, dtype=np.float64)
    autocorr /= overlap
    if autocorr[0] <= 0:
        return None
    autocorr /= autocorr[0]

    # A crop-row period must exceed the observed ridge width.  The bound is
    # therefore scene-derived, not a fixed number in meters or pixels.
    minimum_lag = max(int(np.ceil(row_width_px)), 1)
    # At least two periods must fit inside the profile; otherwise a distant
    # boundary coincidence can masquerade as periodicity.
    maximum_lag = len(autocorr) // 2
    if minimum_lag >= maximum_lag:
        return None
    candidate_lags, properties = find_peaks(
        autocorr[minimum_lag : maximum_lag + 1], prominence=0.0
    )
    candidate_lags = candidate_lags + minimum_lag
    if len(candidate_lags) == 0:
        return None
    peak_scores = np.clip(autocorr[candidate_lags], 0.0, None)
    prominence = properties.get("prominences")
    if prominence is not None and len(prominence) == len(peak_scores):
        peak_scores *= np.maximum(prominence, 0.0)

    # Reward a candidate when its multiples also recur.  This suppresses a
    # strong texture edge that is not periodic across the ROI.
    harmonic_scores = np.zeros_like(peak_scores)
    for index, lag in enumerate(candidate_lags):
        multiples = np.arange(lag, len(autocorr), lag)
        if len(multiples):
            harmonic_scores[index] = np.mean(np.clip(autocorr[multiples], 0.0, None))
    combined = peak_scores * (1.0 + harmonic_scores)
    if not np.any(combined > 0):
        return None

    # Prefer the fundamental period over an equally plausible harmonic.
    best_score = float(combined.max())
    plausible = candidate_lags[combined >= np.nextafter(best_score, -np.inf)]
    best_lag = int(plausible.min())

    line_peaks, _ = find_peaks(clean, prominence=max(spread / 2.0, np.finfo(float).eps))
    differences = np.diff(line_peaks).astype(np.float64)
    if len(differences):
        close = np.abs(differences - best_lag) <= max(row_width_px, np.finfo(float).eps)
        samples_px = differences[close]
    else:
        samples_px = np.empty(0, dtype=np.float64)
    repeated_observations = len(samples_px)
    if repeated_observations == 0:
        samples_px = np.array([float(best_lag)])

    median_px = float(np.median(samples_px))
    mad_px = float(np.median(np.abs(samples_px - median_px)))
    entropy = _robust_entropy(combined)
    period_strength = float(np.clip(autocorr[best_lag], 0.0, 1.0))
    repeatability = 0.0
    if repeated_observations >= 2:
        repeatability = float(np.clip(1.0 - mad_px / max(median_px, 1e-12), 0.0, 1.0))
    confidence = float(np.clip((period_strength + repeatability + (1.0 - entropy)) / 3.0, 0.0, 1.0))
    return SpacingEstimate(
        median_m=median_px * pixel_size_m,
        mad_m=mad_px * pixel_size_m,
        entropy=entropy,
        confidence=confidence,
        samples_m=tuple(float(value * pixel_size_m) for value in samples_px),
    )


def estimate_spacing(
    center_probability: np.ndarray,
    orientation_rad: np.ndarray | float,
    pixel_size_m: float,
    orientation_confidence: np.ndarray | None = None,
    block_size_px: int | None = None,
    reliability_threshold: float = 0.25,
) -> SpacingField:
    """Estimate local row spacing from transverse periodicity.

    If ``block_size_px`` is omitted, a global posterior is returned.  A block
    size can be chosen from the model's effective receptive field to obtain a
    spatially varying spacing distribution.
    """

    center = np.asarray(center_probability, dtype=np.float64)
    if center.ndim != 2:
        raise ValueError("center_probability must be two-dimensional")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive and finite")
    if not 0 <= reliability_threshold <= 1:
        raise ValueError("reliability_threshold must lie between zero and one")
    if np.ndim(orientation_rad) == 0:
        theta_map = np.full(center.shape, float(orientation_rad), dtype=np.float64)
    else:
        theta_map = np.asarray(orientation_rad, dtype=np.float64)
        if theta_map.shape != center.shape:
            raise ValueError("orientation_rad must be scalar or match center_probability")
    if orientation_confidence is not None:
        confidence = np.asarray(orientation_confidence, dtype=np.float64)
        if confidence.shape != center.shape:
            raise ValueError("orientation_confidence must match center_probability")
    else:
        confidence = None

    if block_size_px is None:
        blocks = [(slice(0, center.shape[0]), slice(0, center.shape[1]))]
    else:
        if block_size_px < 3:
            raise ValueError("block_size_px must be at least three")
        stride = max(block_size_px // 2, 1)
        row_starts = list(range(0, max(center.shape[0] - block_size_px + 1, 1), stride))
        col_starts = list(range(0, max(center.shape[1] - block_size_px + 1, 1), stride))
        row_starts.append(max(center.shape[0] - block_size_px, 0))
        col_starts.append(max(center.shape[1] - block_size_px, 0))
        blocks = [
            (
                slice(row, min(row + block_size_px, center.shape[0])),
                slice(col, min(col + block_size_px, center.shape[1])),
            )
            for row in sorted(set(row_starts))
            for col in sorted(set(col_starts))
        ]

    estimates: list[SpacingEstimate] = []
    for row_slice, col_slice in blocks:
        block = center[row_slice, col_slice]
        block_conf = confidence[row_slice, col_slice] if confidence is not None else None
        block_theta = theta_map[row_slice, col_slice]
        axial_s = np.sin(2.0 * block_theta)
        axial_c = np.cos(2.0 * block_theta)
        theta = 0.5 * np.arctan2(
            np.average(axial_s, weights=block_conf)
            if block_conf is not None and block_conf.sum()
            else axial_s.mean(),
            np.average(axial_c, weights=block_conf)
            if block_conf is not None and block_conf.sum()
            else axial_c.mean(),
        )
        profile = _transverse_profile(block, float(theta), block_conf)
        estimate = _estimate_profile_spacing(profile, pixel_size_m)
        if estimate is not None:
            center_row = (row_slice.start + row_slice.stop - 1) / 2.0
            center_col = (col_slice.start + col_slice.stop - 1) / 2.0
            estimates.append(
                SpacingEstimate(
                    median_m=estimate.median_m,
                    mad_m=estimate.mad_m,
                    entropy=estimate.entropy,
                    confidence=estimate.confidence,
                    samples_m=estimate.samples_m,
                    position_xy=(center_col * pixel_size_m, center_row * pixel_size_m),
                )
            )
    reliable = bool(estimates) and (
        float(np.median([item.confidence for item in estimates])) >= reliability_threshold
    )
    return SpacingField(estimates=estimates, reliable=reliable)
