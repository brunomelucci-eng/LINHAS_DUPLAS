"""Topology-aware and uncertainty-ready losses for RowGraphNet.

Every public loss accepts an optional validity mask. Empty masks deliberately
return a differentiable zero rather than NaN, which is important for tiles
containing only nodata or pixels outside the ROI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn

Reduction = Literal["none", "mean", "sum"]


def _binary_target_like(target: Tensor, reference: Tensor) -> Tensor:
    target = target.to(device=reference.device, dtype=reference.dtype)
    if target.ndim == reference.ndim - 1 and reference.ndim >= 3 and reference.shape[1] == 1:
        target = target.unsqueeze(1)
    if target.shape != reference.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match prediction "
            f"shape {tuple(reference.shape)}"
        )
    return target


def _mask_like(mask: Tensor | None, reference: Tensor) -> Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device=reference.device)
    if mask.ndim == reference.ndim - 1 and reference.ndim >= 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != reference.ndim:
        raise ValueError("valid_mask must have the prediction rank or omit its channel axis")
    try:
        mask = torch.broadcast_to(mask, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask shape {tuple(mask.shape)} is not broadcastable to {tuple(reference.shape)}"
        ) from exc
    return mask.to(dtype=reference.dtype)


def _masked_reduce(values: Tensor, mask: Tensor | None, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values if mask is None else values * mask
    if reduction == "sum":
        return values.sum() if mask is None else (values * mask).sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if values.numel() == 0:
        return values.sum() * 0.0
    if mask is None:
        return values.mean()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _combine_masks(reference: Tensor, *masks: Tensor | None) -> Tensor | None:
    """Intersect binary supervision masks on ``reference``'s grid."""

    prepared = [_mask_like(mask, reference) for mask in masks if mask is not None]
    if not prepared:
        return None
    combined = torch.ones_like(reference)
    for mask in prepared:
        assert mask is not None
        combined = combined * (mask > 0).to(dtype=reference.dtype)
    return combined


def _normalize_row_id_mask(row_id_mask: Tensor) -> Tensor:
    """Return integer instance labels as BHW, accepting BHW or B1HW input."""

    if row_id_mask.ndim == 2:
        row_id_mask = row_id_mask.unsqueeze(0)
    elif row_id_mask.ndim == 4 and row_id_mask.shape[1] == 1:
        row_id_mask = row_id_mask[:, 0]
    if row_id_mask.ndim != 3:
        raise ValueError("row_id_mask must have shape HW, BHW, or B1HW")
    if row_id_mask.is_complex():
        raise ValueError("row_id_mask must contain real non-negative integer labels")
    if row_id_mask.is_floating_point():
        if not bool(torch.isfinite(row_id_mask).all().detach().cpu()):
            raise ValueError("row_id_mask must contain finite labels")
        rounded = row_id_mask.round()
        if not bool(torch.eq(row_id_mask, rounded).all().detach().cpu()):
            raise ValueError("row_id_mask must contain integer-valued labels")
        row_id_mask = rounded
    row_id_mask = row_id_mask.to(dtype=torch.int64)
    if bool((row_id_mask < 0).any().detach().cpu()):
        raise ValueError("row_id_mask labels must be non-negative")
    return row_id_mask


def _corridor_weights_for_sample(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Build scale-free soft Voronoi corridors for one label image."""

    labels = labels.copy()
    labels[~valid] = 0
    instance_ids = np.unique(labels[labels > 0])
    if instance_ids.size < 2:
        return np.zeros(labels.shape, dtype=np.float32)

    # Each layer contains the Euclidean pixel distance to one annotated row.
    # Pixel units are intentional: the expression below divides by the
    # measured separation of the same pair, so no agronomic distance enters.
    distances = np.stack(
        [distance_transform_edt(labels != instance_id) for instance_id in instance_ids]
    ).astype(np.float32, copy=False)
    count = int(instance_ids.size)
    separation = np.zeros((count, count), dtype=np.float32)
    for first in range(count):
        first_pixels = labels == instance_ids[first]
        for second in range(first + 1, count):
            second_pixels = labels == instance_ids[second]
            pair_distance = min(
                float(distances[first][second_pixels].min()),
                float(distances[second][first_pixels].min()),
            )
            separation[first, second] = pair_distance
            separation[second, first] = pair_distance

    nearest_indices = np.argpartition(distances, kth=1, axis=0)[:2]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=0)
    first_distance = nearest_distances[0]
    second_distance = nearest_distances[1]
    distance_sum = first_distance + second_distance
    pair_separation = separation[nearest_indices[0], nearest_indices[1]]

    # ``balance`` peaks halfway between two rows. ``between_ratio`` is one on
    # the shortest corridor joining that pair and decays beyond their ends.
    # Both are dimensionless and adapt to the geometry present in this tile.
    safe_sum = np.maximum(distance_sum, np.finfo(np.float32).eps)
    balance = 4.0 * first_distance * second_distance / np.square(safe_sum)
    between_ratio = np.clip(pair_separation / safe_sum, 0.0, 1.0)
    weights = balance * np.square(between_ratio)
    weights[(labels > 0) | (~valid)] = 0.0
    weights[~np.isfinite(weights)] = 0.0
    return weights.astype(np.float32, copy=False)


def instance_corridor_weights(
    row_id_mask: Tensor,
    *,
    valid_mask: Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Derive soft inter-row corridors from instance labels.

    ``row_id_mask`` uses zero for background and a positive integer per row.
    The returned B1HW tensor is largest where a background pixel is equally
    close to two different rows and lies on their shortest connecting
    corridor. Distances are normalized by the measured separation of that
    exact row pair; consequently this target contains no fixed pixel, metre,
    spacing, or angle threshold. Invalid/ROI-exterior/nodata pixels receive
    zero weight.

    Corridor construction is target-only and therefore need not carry a
    gradient. The loss consuming these weights remains differentiable with
    respect to predicted centerline logits.
    """

    labels = _normalize_row_id_mask(row_id_mask)
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("corridor dtype must be floating point")
    reference = torch.zeros(
        (labels.shape[0], 1, labels.shape[1], labels.shape[2]),
        device=labels.device,
        dtype=dtype,
    )
    if valid_mask is None:
        valid = torch.ones_like(reference, dtype=torch.bool)
    else:
        if valid_mask.ndim == 2 and labels.shape[0] == 1:
            valid_mask = valid_mask.unsqueeze(0)
        prepared_valid = _mask_like(valid_mask, reference)
        assert prepared_valid is not None
        valid = prepared_valid > 0

    labels_numpy = labels.detach().cpu().numpy()
    valid_numpy = valid[:, 0].detach().cpu().numpy()
    weights = np.stack(
        [
            _corridor_weights_for_sample(labels_numpy[index], valid_numpy[index])
            for index in range(labels.shape[0])
        ],
        axis=0,
    )[:, None]
    return torch.as_tensor(weights, device=labels.device, dtype=dtype)


def instance_bridge_loss(
    prediction: Tensor,
    row_id_mask: Tensor,
    *,
    valid_mask: Tensor | None = None,
    corridor_weights: Tensor | None = None,
    from_logits: bool = True,
    reduction: Reduction = "mean",
    epsilon: float = 1e-7,
) -> Tensor:
    """Penalize center probability in scale-free corridors between rows.

    This is an instance-aware *negative* objective: pixels belonging to a row
    are excluded, while background pixels between two different positive
    ``row_id`` labels receive a geometry-derived weight. It therefore opposes
    a bridge to a neighbouring row without suppressing a gap inside the same
    ``row_id``. ``valid_mask`` must already exclude ignore, outside-ROI and
    nodata pixels.

    ``corridor_weights`` may be cached by a dataset to avoid recomputing the
    target-only distance transforms. Empty masks and tiles with fewer than two
    row instances return a differentiable zero.
    """

    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("center prediction must have shape [batch, 1, height, width]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    labels = _normalize_row_id_mask(row_id_mask)
    expected = (prediction.shape[0], *prediction.shape[-2:])
    if tuple(labels.shape) != expected:
        raise ValueError(
            f"row_id_mask shape {tuple(labels.shape)} does not match expected {expected}"
        )
    mask = _mask_like(valid_mask, prediction)
    if corridor_weights is None:
        weights = instance_corridor_weights(
            labels,
            valid_mask=mask,
            dtype=prediction.dtype,
        ).to(device=prediction.device)
    else:
        weights = _mask_like(corridor_weights, prediction)
        assert weights is not None
        if not bool(torch.isfinite(weights).all().detach().cpu()):
            raise ValueError("corridor_weights must be finite")
        if bool((weights < 0).any().detach().cpu()):
            raise ValueError("corridor_weights must be non-negative")
    if mask is not None:
        weights = weights * (mask > 0).to(dtype=prediction.dtype)

    if from_logits:
        values = F.softplus(prediction)  # stable BCE(logits, target=0)
    else:
        if bool(((prediction < 0) | (prediction > 1)).any().detach().cpu()):
            raise ValueError("probability prediction must lie in [0, 1]")
        values = -torch.log1p(-prediction.clamp(max=1.0 - epsilon))
    weighted = values * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    return weighted.sum() / weights.sum().clamp_min(1.0)


def binary_focal_loss_with_logits(
    logits: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Numerically stable binary focal loss operating on logits."""

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1] or None")
    target = _binary_target_like(target, logits)
    mask = _mask_like(valid_mask, logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    values = cross_entropy * (1.0 - probability_true).pow(gamma)
    if alpha is not None:
        alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)
        values = values * alpha_factor
    return _masked_reduce(values, mask, reduction)


def soft_dice_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    from_logits: bool = True,
    smooth: float = 1.0,
    epsilon: float = 1e-7,
) -> Tensor:
    """Batch-averaged soft Dice loss with a well-defined empty-mask result."""

    if smooth < 0 or epsilon <= 0:
        raise ValueError("smooth must be non-negative and epsilon positive")
    probability = torch.sigmoid(prediction) if from_logits else prediction
    target = _binary_target_like(target, probability)
    mask = _mask_like(valid_mask, probability)
    if mask is not None:
        probability = probability * mask
        target = target * mask
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    score = (2.0 * intersection + smooth) / (denominator + smooth + epsilon)
    per_sample = 1.0 - score
    if mask is None:
        return per_sample.mean()
    active = (mask.sum(dim=dimensions) > 0).to(dtype=per_sample.dtype)
    return (per_sample * active).sum() / active.sum().clamp_min(1.0)


def _soft_erode(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError("soft morphology expects BCHW tensors")
    erosion_height = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erosion_width = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(erosion_height, erosion_width)


def _soft_dilate(image: Tensor) -> Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_open(image: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeletonize(image: Tensor, *, iterations: int = 10) -> Tensor:
    """Differentiable approximation of morphological skeletonization."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    eroded = image
    for _ in range(iterations):
        eroded = _soft_erode(eroded)
        opened = _soft_open(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    from_logits: bool = True,
    iterations: int = 10,
    smooth: float = 1.0,
    epsilon: float = 1e-7,
) -> Tensor:
    """Soft-clDice loss for connectivity of thin centerline structures."""

    probability = torch.sigmoid(prediction) if from_logits else prediction
    target = _binary_target_like(target, probability)
    mask = _mask_like(valid_mask, probability)
    if mask is not None:
        probability = probability * mask
        target = target * mask

    predicted_skeleton = soft_skeletonize(probability, iterations=iterations)
    target_skeleton = soft_skeletonize(target, iterations=iterations)
    dimensions = tuple(range(1, probability.ndim))
    topology_precision = ((predicted_skeleton * target).sum(dim=dimensions) + smooth) / (
        predicted_skeleton.sum(dim=dimensions) + smooth + epsilon
    )
    topology_sensitivity = ((target_skeleton * probability).sum(dim=dimensions) + smooth) / (
        target_skeleton.sum(dim=dimensions) + smooth + epsilon
    )
    cldice = (2.0 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + epsilon
    )
    per_sample = 1.0 - cldice
    if mask is None:
        return per_sample.mean()
    active = (mask.sum(dim=dimensions) > 0).to(dtype=per_sample.dtype)
    return (per_sample * active).sum() / active.sum().clamp_min(1.0)


def _normalize_axial(vector: Tensor, epsilon: float) -> Tensor:
    if vector.ndim < 2 or vector.shape[1] != 2:
        raise ValueError("axial orientation vectors must have two channels [sin(2t), cos(2t)]")
    norm = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    normalized = vector / norm.clamp_min(epsilon)
    fallback = torch.zeros_like(vector)
    fallback[:, 1:2] = 1.0
    return torch.where(norm > epsilon, normalized, fallback)


def axial_orientation_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    target_is_angle: bool = False,
    reduction: Reduction = "mean",
    epsilon: float = 1e-6,
) -> Tensor:
    """Circular axial loss ``1 - cos(2 * (theta_pred - theta_gt))``.

    By default both tensors use two channels ordered as ``sin(2 theta)`` and
    ``cos(2 theta)``. With ``target_is_angle=True``, ``target`` contains theta
    in radians and may have shape BHW or B1HW.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    predicted_vector = _normalize_axial(prediction, epsilon)
    if target_is_angle:
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if target.ndim == prediction.ndim - 1:
            target = target.unsqueeze(1)
        expected = (prediction.shape[0], 1, *prediction.shape[2:])
        if target.shape != expected:
            raise ValueError(f"angle target must have shape {expected}")
        target_vector = torch.cat((torch.sin(2.0 * target), torch.cos(2.0 * target)), dim=1)
    else:
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if target.shape != prediction.shape:
            raise ValueError("orientation target must match the prediction shape")
        target_vector = _normalize_axial(target, epsilon)

    cosine_double_delta = (predicted_vector * target_vector).sum(dim=1, keepdim=True)
    values = 1.0 - cosine_double_delta.clamp(-1.0, 1.0)
    mask = _mask_like(valid_mask, values)
    return _masked_reduce(values, mask, reduction)


def masked_smooth_l1_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    beta: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Smooth-L1 with safe reduction for absent supervision."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.ndim == prediction.ndim - 1 and prediction.ndim >= 3 and prediction.shape[1] == 1:
        target = target.unsqueeze(1)
    if target.shape != prediction.shape:
        raise ValueError("regression target must match prediction shape")
    mask = _mask_like(valid_mask, prediction)
    values = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return _masked_reduce(values, mask, reduction)


class RowGraphNetLoss(nn.Module):
    """Compose dense, topology, orientation, repair and graph objectives.

    The returned mapping contains each available task loss plus ``total``.
    Optional homoscedastic weighting learns one log variance per task. A task
    with no valid elements is excluded from both terms, preventing its learned
    weight from drifting on empty ROI tiles.
    """

    task_names = (
        "band",
        "center",
        "topology",
        "instance_bridge",
        "orientation",
        "endpoints",
        "distance",
        "uncertainty",
        "repair",
        "graph",
    )

    def __init__(
        self,
        *,
        learned_weighting: bool = True,
        static_weights: Mapping[str, float] | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cldice_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.learned_weighting = bool(learned_weighting)
        self.static_weights = {name: 1.0 for name in self.task_names}
        if static_weights:
            unknown = set(static_weights) - set(self.task_names)
            if unknown:
                raise ValueError(f"unknown task weights: {sorted(unknown)}")
            self.static_weights.update({key: float(value) for key, value in static_weights.items()})
        if any(value < 0 for value in self.static_weights.values()):
            raise ValueError("static task weights cannot be negative")
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.cldice_iterations = int(cldice_iterations)
        if self.cldice_iterations < 1:
            raise ValueError("cldice_iterations must be positive")
        self.log_variances = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.task_names}
        )
        if not self.learned_weighting:
            for parameter in self.log_variances.values():
                parameter.requires_grad_(False)

    def forward(
        self,
        predictions: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Evaluate all available objectives using the center-target contract.

        Preferred center targets are:

        - ``center_observed``: positives supported directly by imagery;
        - ``center_complete``: the full annotated row, including occlusion;
        - ``row_id_mask``: zero background and one positive integer per row;
        - ``valid_mask``: excludes ignore, outside-ROI and nodata pixels.

        Focal + Dice use ``center_observed_logits``/``center_observed`` and
        automatically ignore pixels that are positive only in
        ``center_complete``. Soft-clDice and instance separation use
        ``center_complete_logits`` with the complete targets. Prediction key
        ``center_logits`` and target key ``center`` remain supported as
        fallbacks for both objectives. Per-objective masks may be supplied as
        ``center_observed_valid_mask`` and ``center_complete_valid_mask``.
        """

        valid_mask = targets.get("valid_mask", valid_mask)
        losses: dict[str, Tensor] = {}
        active: dict[str, bool] = {}

        if "band" in targets and "band_logits" in predictions:
            band_mask = targets.get("band_valid_mask", valid_mask)
            losses["band"] = self._binary_dense_loss(
                predictions["band_logits"], targets["band"], band_mask
            )
            active["band"] = self._has_support(band_mask, predictions["band_logits"])

        center_observed = targets.get("center_observed")
        center_complete = targets.get("center_complete")
        legacy_center = targets.get("center")
        if center_observed is None and legacy_center is not None:
            center_observed = legacy_center
        if center_complete is None and legacy_center is not None:
            center_complete = legacy_center

        legacy_center_logits = predictions.get("center_logits")
        observed_logits = predictions.get("center_observed_logits")
        complete_logits = predictions.get("center_complete_logits")
        if observed_logits is None:
            observed_logits = legacy_center_logits
        if complete_logits is None:
            complete_logits = legacy_center_logits
        if observed_logits is None:
            observed_logits = complete_logits
        if complete_logits is None:
            complete_logits = observed_logits

        if (observed_logits is not None or complete_logits is not None) and (
            center_observed is not None or center_complete is not None
        ):
            complete_mask = (
                _combine_masks(
                    complete_logits,
                    valid_mask,
                    targets.get("center_complete_valid_mask"),
                )
                if complete_logits is not None
                else None
            )
            observed_mask = (
                _combine_masks(
                    observed_logits,
                    valid_mask,
                    targets.get("center_observed_valid_mask"),
                )
                if observed_logits is not None
                else None
            )

            if center_observed is not None and observed_logits is not None:
                # Complete-only positives are occluded centerline supervision,
                # never negative examples for the observed-center objective.
                occluded = targets.get("occluded_mask")
                if occluded is None:
                    occluded = targets.get("center_occluded")
                observed_target = _binary_target_like(center_observed, observed_logits)
                if center_complete is not None:
                    complete_target = _binary_target_like(center_complete, observed_logits)
                    inferred_occluded = (complete_target > 0) & (observed_target <= 0)
                    occluded = (
                        inferred_occluded
                        if occluded is None
                        else inferred_occluded | (_mask_like(occluded, observed_logits) > 0)
                    )
                if occluded is not None:
                    prepared_occluded = _mask_like(occluded, observed_logits)
                    assert prepared_occluded is not None
                    observed_mask = _combine_masks(
                        observed_logits,
                        observed_mask,
                        prepared_occluded <= 0,
                    )
                losses["center"] = self._binary_dense_loss(
                    observed_logits,
                    observed_target,
                    observed_mask,
                )
                active["center"] = self._has_support(observed_mask, observed_logits)

            if center_complete is not None and complete_logits is not None:
                losses["topology"] = soft_cldice_loss(
                    complete_logits,
                    center_complete,
                    valid_mask=complete_mask,
                    iterations=self.cldice_iterations,
                )
                active["topology"] = self._has_support(complete_mask, complete_logits)

            if "row_id_mask" in targets and complete_logits is not None:
                cached_corridor = targets.get("instance_corridor_weights")
                if cached_corridor is None:
                    corridor = instance_corridor_weights(
                        targets["row_id_mask"],
                        valid_mask=complete_mask,
                        dtype=complete_logits.dtype,
                    ).to(device=complete_logits.device)
                else:
                    corridor = _mask_like(cached_corridor, complete_logits)
                    assert corridor is not None
                    if complete_mask is not None:
                        corridor = corridor * (complete_mask > 0).to(corridor.dtype)
                losses["instance_bridge"] = instance_bridge_loss(
                    complete_logits,
                    targets["row_id_mask"],
                    valid_mask=complete_mask,
                    corridor_weights=corridor,
                )
                active["instance_bridge"] = bool(torch.count_nonzero(corridor).detach().cpu())

        if "orientation" in targets and "orientation" in predictions:
            orientation_mask = targets.get("orientation_valid_mask", valid_mask)
            losses["orientation"] = axial_orientation_loss(
                predictions["orientation"],
                targets["orientation"],
                valid_mask=orientation_mask,
            )
            active["orientation"] = self._has_support(
                orientation_mask, predictions["orientation"][:, :1]
            )

        if "endpoints" in targets and "endpoints_logits" in predictions:
            endpoint_mask = targets.get("endpoints_valid_mask", valid_mask)
            losses["endpoints"] = binary_focal_loss_with_logits(
                predictions["endpoints_logits"],
                targets["endpoints"],
                valid_mask=endpoint_mask,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            active["endpoints"] = self._has_support(endpoint_mask, predictions["endpoints_logits"])

        if "distance" in targets and "distance" in predictions:
            distance_mask = targets.get("distance_valid_mask", valid_mask)
            distance_prediction = predictions["distance"]
            distance_target = targets["distance"]
            losses["distance"] = masked_smooth_l1_loss(
                distance_prediction,
                distance_target,
                valid_mask=distance_mask,
            )
            distance_active = self._has_support(distance_mask, distance_prediction)
            active["distance"] = distance_active
            if "uncertainty" in predictions:
                prepared_target = distance_target.to(
                    device=distance_prediction.device,
                    dtype=distance_prediction.dtype,
                )
                if prepared_target.ndim == distance_prediction.ndim - 1:
                    prepared_target = prepared_target.unsqueeze(1)
                absolute_error = (distance_prediction - prepared_target).detach().abs()
                losses["uncertainty"] = masked_smooth_l1_loss(
                    predictions["uncertainty"],
                    absolute_error,
                    valid_mask=distance_mask,
                )
                active["uncertainty"] = distance_active

        if "repair" in targets and "repair_logits" in predictions:
            repair_mask = targets.get("repair_valid_mask", valid_mask)
            losses["repair"] = self._binary_dense_loss(
                predictions["repair_logits"], targets["repair"], repair_mask
            )
            active["repair"] = self._has_support(repair_mask, predictions["repair_logits"])

        if "edge_labels" in targets and "edge_logits" in predictions:
            edge_mask = targets.get("edge_valid_mask")
            losses["graph"] = binary_focal_loss_with_logits(
                predictions["edge_logits"],
                targets["edge_labels"],
                valid_mask=edge_mask,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            active["graph"] = self._has_support(edge_mask, predictions["edge_logits"])

        anchor = next(
            (value for value in predictions.values() if isinstance(value, Tensor)),
            None,
        )
        if anchor is None:
            raise ValueError("predictions must contain at least one tensor")
        total = anchor.sum() * 0.0
        for name, loss in losses.items():
            weight = self.static_weights[name]
            if not active.get(name, False) or weight == 0:
                continue
            weighted_loss = loss * weight
            if self.learned_weighting:
                log_variance = self.log_variances[name]
                total = total + torch.exp(-log_variance) * weighted_loss + log_variance
            else:
                total = total + weighted_loss
        losses["total"] = total
        return losses

    def _binary_dense_loss(
        self,
        logits: Tensor,
        target: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        return binary_focal_loss_with_logits(
            logits,
            target,
            valid_mask=valid_mask,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ) + soft_dice_loss(logits, target, valid_mask=valid_mask)

    @staticmethod
    def _has_support(mask: Tensor | None, reference: Tensor) -> bool:
        if reference.numel() == 0:
            return False
        if mask is None:
            return True
        prepared = _mask_like(mask, reference)
        assert prepared is not None
        return bool(torch.count_nonzero(prepared).detach().cpu())


# Descriptive alias retained for callers that prefer the architecture-neutral name.
MultiTaskLoss = RowGraphNetLoss


__all__ = [
    "MultiTaskLoss",
    "RowGraphNetLoss",
    "axial_orientation_loss",
    "binary_focal_loss_with_logits",
    "instance_bridge_loss",
    "instance_corridor_weights",
    "masked_smooth_l1_loss",
    "soft_cldice_loss",
    "soft_dice_loss",
    "soft_skeletonize",
]
