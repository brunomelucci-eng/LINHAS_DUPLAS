"""Axial orientation estimation and automatic direction-regime discovery.

Angles are represented by ``(sin(2 theta), cos(2 theta))``. Doubling the
angle makes opposite tangent directions identical, which is the correct
statistics for an unoriented crop row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class FusedOrientationField:
    sin2: np.ndarray
    cos2: np.ndarray
    confidence: np.ndarray

    @property
    def theta_rad(self) -> np.ndarray:
        return 0.5 * np.arctan2(self.sin2, self.cos2)


@dataclass(slots=True, frozen=True)
class OrientationRegime:
    regime_id: int
    orientation_rad: float
    axial_concentration: float
    weight: float
    sample_count: int

    @property
    def orientation_deg(self) -> float:
        return float(np.degrees(self.orientation_rad) % 180.0)


def pixel_theta_to_world(theta_rad: np.ndarray | float, transform: Affine) -> np.ndarray:
    """Transform axial raster angles into the world-CRS Cartesian basis.

    Raster angles use ``x=column`` and ``y=row`` (positive downward).  Vector
    geometry uses the CRS x/y axes.  Applying only a sign flip is incorrect for
    rotated, sheared, or non-square pixels, so the complete linear part of the
    affine transform is used here.  Translation is intentionally irrelevant.
    """

    theta = np.asarray(theta_rad, dtype=np.float64)
    pixel_x = np.cos(theta)
    pixel_y = np.sin(theta)
    world_x = transform.a * pixel_x + transform.b * pixel_y
    world_y = transform.d * pixel_x + transform.e * pixel_y
    norm = np.hypot(world_x, world_y)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("affine linear basis cannot collapse an orientation vector")
    return np.mod(np.arctan2(world_y / norm, world_x / norm), np.pi)


def orientation_field_to_world(
    field: FusedOrientationField,
    transform: Affine,
) -> FusedOrientationField:
    """Return an equivalent axial field expressed in world x/y coordinates."""

    theta_world = pixel_theta_to_world(field.theta_rad, transform)
    return FusedOrientationField(
        sin2=np.sin(2.0 * theta_world),
        cos2=np.cos(2.0 * theta_world),
        confidence=np.asarray(field.confidence, dtype=np.float64).copy(),
    )


def _as_unit_axial(sin2: np.ndarray, cos2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.hypot(sin2, cos2)
    safe = np.maximum(norm, np.finfo(np.float64).eps)
    return sin2 / safe, cos2 / safe


def structure_tensor_orientation(
    center_probability: np.ndarray,
    sigma_px: float,
    valid_mask: np.ndarray | None = None,
) -> FusedOrientationField:
    """Estimate ridge tangents from the local structure tensor.

    ``sigma_px`` is a scale measured or calibrated in pixels (normally from
    predicted row width); it is intentionally not an agronomic distance.
    """

    image = np.asarray(center_probability, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("center_probability must be a 2-D array")
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("sigma_px must be positive and finite")

    clean = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    valid: np.ndarray | None = None
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != image.shape:
            raise ValueError("valid_mask must match center_probability")
        clean[~valid] = 0.0

    grad_y = gaussian_filter(clean, sigma=sigma_px, order=(1, 0), mode="reflect")
    grad_x = gaussian_filter(clean, sigma=sigma_px, order=(0, 1), mode="reflect")
    del clean

    smooth_sigma = max(float(sigma_px), float(np.finfo(np.float32).eps))
    j_xx = gaussian_filter(grad_x * grad_x, smooth_sigma, mode="reflect")
    j_xy = gaussian_filter(grad_x * grad_y, smooth_sigma, mode="reflect")
    del grad_x
    j_yy = gaussian_filter(grad_y * grad_y, smooth_sigma, mode="reflect")
    del grad_y

    gradient_angle = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy)
    tangent = gradient_angle + (np.pi / 2.0)
    sin2 = np.sin(2.0 * tangent).astype(np.float32)
    cos2 = np.cos(2.0 * tangent).astype(np.float32)
    del gradient_angle, tangent

    diff = j_xx - j_yy
    discriminant = np.sqrt(np.maximum(diff * diff + 4.0 * (j_xy * j_xy), 0.0))
    del diff, j_xy

    trace = j_xx + j_yy
    del j_xx, j_yy

    eps = float(np.finfo(np.float32).eps)
    confidence = discriminant / np.maximum(trace, eps)
    del discriminant

    # Amostragem representativa para percentil se imagem for grande (>1M pixels)
    if np.any(trace > 0):
        if trace.size > 1_000_000:
            step = max(1, int(math.sqrt(trace.size / 500_000)))
            sample = trace[::step, ::step]
            pos = sample[sample > 0]
            energy_scale = float(np.percentile(pos, 95)) if len(pos) > 0 else 0.0
        else:
            pos = trace[trace > 0]
            energy_scale = float(np.percentile(pos, 95)) if len(pos) > 0 else 0.0
    else:
        energy_scale = 0.0

    if energy_scale > 0:
        confidence *= np.clip(trace / energy_scale, 0.0, 1.0)
    else:
        confidence.fill(0.0)
    del trace

    if valid is not None:
        confidence[~valid] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    return FusedOrientationField(sin2, cos2, confidence)


def fuse_orientation_fields(
    learned_sin2: np.ndarray,
    learned_cos2: np.ndarray,
    learned_uncertainty: np.ndarray,
    tensor_field: FusedOrientationField,
    valid_mask: np.ndarray | None = None,
) -> FusedOrientationField:
    """Fuse learned and image-derived axial vectors by their confidence."""

    ls = np.asarray(learned_sin2, dtype=np.float32)
    lc = np.asarray(learned_cos2, dtype=np.float32)
    uncertainty = np.asarray(learned_uncertainty, dtype=np.float32)
    if not (ls.shape == lc.shape == uncertainty.shape == tensor_field.sin2.shape):
        raise ValueError("all orientation maps must have the same shape")
    valid = (
        np.ones(ls.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != ls.shape:
        raise ValueError("valid_mask must match orientation maps")
    if np.any(valid & (~np.isfinite(ls) | ~np.isfinite(lc) | ~np.isfinite(uncertainty))):
        raise ValueError("learned orientation and uncertainty must be finite in valid pixels")
    ls[~valid | ~np.isfinite(ls)] = 0.0
    lc[~valid | ~np.isfinite(lc)] = 1.0
    uncertainty[~valid | ~np.isfinite(uncertainty)] = np.inf
    ls, lc = _as_unit_axial(ls, lc)
    ts, tc = _as_unit_axial(tensor_field.sin2, tensor_field.cos2)

    if np.any(uncertainty < 0):
        raise ValueError("learned_uncertainty must be non-negative")
    # The network predicts a non-negative aleatoric scale, not a probability.
    # Map that scale monotonically to confidence without imposing an arbitrary
    # upper cutoff: zero uncertainty has unit weight and an infinite scale has
    # zero weight.  This also handles invalid pixels marked with ``inf``.
    learned_weight = np.divide(
        1.0,
        1.0 + uncertainty,
        out=np.zeros_like(uncertainty),
        where=np.isfinite(uncertainty),
    )
    tensor_weight = np.clip(tensor_field.confidence, 0.0, 1.0)
    sum_s = learned_weight * ls + tensor_weight * ts
    sum_c = learned_weight * lc + tensor_weight * tc
    fused_s, fused_c = _as_unit_axial(sum_s, sum_c)
    resultant = np.hypot(sum_s, sum_c)
    confidence = resultant / np.maximum(learned_weight + tensor_weight, float(np.finfo(np.float32).eps))
    confidence *= np.maximum(learned_weight, tensor_weight)

    zero_evidence = (learned_weight + tensor_weight) <= float(np.finfo(np.float32).eps)
    confidence[zero_evidence] = 0.0
    fused_s[zero_evidence] = 0.0
    fused_c[zero_evidence] = 1.0
    if valid_mask is not None:
        confidence[~valid] = 0.0
    return FusedOrientationField(fused_s.astype(np.float32), fused_c.astype(np.float32), np.clip(confidence, 0.0, 1.0).astype(np.float32))


def _weighted_axial_kmeans(
    vectors: np.ndarray,
    weights: np.ndarray,
    k: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic k-means on the doubled-angle unit circle."""

    first = int(np.argmax(weights))
    centers = [vectors[first]]
    for _ in range(1, k):
        similarities = vectors @ np.stack(centers).T
        nearest_distance = 1.0 - similarities.max(axis=1)
        score = nearest_distance * weights
        centers.append(vectors[int(np.argmax(score))])
    center_array = np.stack(centers)

    labels = np.zeros(len(vectors), dtype=np.int32)
    for _ in range(max_iter):
        previous = labels.copy()
        labels = np.argmax(vectors @ center_array.T, axis=1).astype(np.int32)
        for idx in range(k):
            member = labels == idx
            if not np.any(member):
                residual = 1.0 - np.max(vectors @ center_array.T, axis=1)
                replacement = int(np.argmax(residual * weights))
                center_array[idx] = vectors[replacement]
                continue
            resultant = np.sum(vectors[member] * weights[member, None], axis=0)
            norm = np.linalg.norm(resultant)
            if norm > np.finfo(float).eps:
                center_array[idx] = resultant / norm
        if np.array_equal(labels, previous):
            break
    residual = 1.0 - np.sum(vectors * center_array[labels], axis=1)
    weighted_sse = float(np.sum(weights * residual**2))
    return labels, center_array, weighted_sse


def discover_orientation_regimes(
    field: FusedOrientationField,
    evidence: np.ndarray | None = None,
    max_regimes: int = 4,
    min_confidence_quantile: float = 0.5,
    max_samples: int = 20_000,
) -> tuple[list[OrientationRegime], np.ndarray]:
    """Choose the number of axial direction regimes with a BIC criterion.

    Sampling and the maximum number of mixture components are computational
    safeguards.  No angle is fixed: component centers and dispersions are
    learned from the current scene.
    """

    if max_regimes < 1:
        raise ValueError("max_regimes must be at least one")
    confidence = np.asarray(field.confidence, dtype=np.float64)
    if evidence is not None:
        ev = np.asarray(evidence, dtype=np.float64)
        if ev.shape != confidence.shape:
            raise ValueError("evidence must match the orientation field")
        confidence = confidence * np.clip(ev, 0.0, 1.0)
    finite_positive = np.isfinite(confidence) & (confidence > 0)
    if not np.any(finite_positive):
        return [], np.full(confidence.shape, -1, dtype=np.int16)
    cutoff = np.quantile(confidence[finite_positive], np.clip(min_confidence_quantile, 0, 1))
    selected = finite_positive & (confidence >= cutoff)
    flat_indices = np.flatnonzero(selected)
    if len(flat_indices) > max_samples:
        # Deterministic uniform subsampling keeps runs reproducible.
        take = np.linspace(0, len(flat_indices) - 1, max_samples, dtype=np.int64)
        flat_indices = flat_indices[take]

    sin_values = field.sin2.ravel()[flat_indices]
    cos_values = field.cos2.ravel()[flat_indices]
    sample_s, sample_c = _as_unit_axial(sin_values, cos_values)
    vectors = np.column_stack((sample_c, sample_s))
    weights = confidence.ravel()[flat_indices]
    weights /= np.maximum(weights.mean(), np.finfo(float).eps)

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    best_bic = np.inf
    upper = min(max_regimes, len(vectors))
    for k in range(1, upper + 1):
        labels, centers, sse = _weighted_axial_kmeans(vectors, weights, k, max_iter=100)
        # A component smaller than the sampling fluctuation bound is an outlier
        # group, not a defensible direction regime.  The bound grows with the
        # actual sample count and introduces no angular tolerance.
        sampling_bound = np.sqrt(len(vectors) * np.log(max(len(vectors), 2)))
        if k > 1 and np.min(np.bincount(labels, minlength=k)) < sampling_bound:
            continue
        variance = max(sse / max(len(vectors), 1), np.finfo(float).eps)
        # Each component contributes an angle, a mass, and shared residual
        # variance.  BIC naturally refuses needless direction regimes.
        parameter_count = 2 * k
        bic = len(vectors) * np.log(variance) + parameter_count * np.log(len(vectors))
        if bic < best_bic:
            best_bic = bic
            best = labels, centers, sse
    assert best is not None
    labels, centers, _ = best

    regimes: list[OrientationRegime] = []
    total_weight = float(np.sum(weights))
    for idx, center in enumerate(centers):
        member = labels == idx
        member_weights = weights[member]
        resultant = np.sum(vectors[member] * member_weights[:, None], axis=0)
        concentration = float(np.linalg.norm(resultant) / np.maximum(member_weights.sum(), 1e-12))
        theta = 0.5 * np.arctan2(center[1], center[0])
        regimes.append(
            OrientationRegime(
                regime_id=idx,
                orientation_rad=float(theta % np.pi),
                axial_concentration=concentration,
                weight=float(member_weights.sum() / total_weight),
                sample_count=int(member.sum()),
            )
        )

    # Assign every pixel with evidence to its closest learned axial direction.
    label_map = np.full(confidence.shape, -1, dtype=np.int16)
    if np.any(finite_positive):
        c_vals = np.asarray(field.cos2)[finite_positive]
        s_vals = np.asarray(field.sin2)[finite_positive]
        norm = np.hypot(c_vals, s_vals)
        safe = np.maximum(norm, np.finfo(np.float64).eps)
        u_c = c_vals / safe
        u_s = s_vals / safe
        best_dots = np.full(len(c_vals), -np.inf, dtype=np.float64)
        best_labels = np.zeros(len(c_vals), dtype=np.int16)
        for c_idx, center in enumerate(centers):
            dot = u_c * center[0] + u_s * center[1]
            better = dot > best_dots
            best_dots[better] = dot[better]
            best_labels[better] = c_idx
        label_map[finite_positive] = best_labels
    regimes.sort(key=lambda item: (-item.weight, item.orientation_rad))
    return regimes, label_map
