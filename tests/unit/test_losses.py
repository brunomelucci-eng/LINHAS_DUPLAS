from __future__ import annotations

import math

import torch

from rowgraphnet.losses import (
    RowGraphNetLoss,
    axial_orientation_loss,
    binary_focal_loss_with_logits,
    instance_bridge_loss,
    instance_corridor_weights,
    masked_smooth_l1_loss,
    soft_cldice_loss,
    soft_dice_loss,
)


def test_all_primitive_losses_are_zero_and_finite_for_empty_valid_mask() -> None:
    logits = torch.randn(2, 1, 8, 9, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    empty_mask = torch.zeros(2, 8, 9, dtype=torch.bool)
    orientation = torch.nn.functional.normalize(torch.randn(2, 2, 8, 9), dim=1)

    losses = [
        binary_focal_loss_with_logits(logits, target, valid_mask=empty_mask),
        soft_dice_loss(logits, target, valid_mask=empty_mask),
        soft_cldice_loss(logits, target, valid_mask=empty_mask, iterations=3),
        instance_bridge_loss(
            logits,
            torch.zeros(2, 8, 9, dtype=torch.int64),
            valid_mask=empty_mask,
        ),
        masked_smooth_l1_loss(logits, target, valid_mask=empty_mask),
        axial_orientation_loss(
            orientation,
            orientation.detach(),
            valid_mask=empty_mask,
        ),
    ]

    for loss in losses:
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    sum(losses).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_axial_orientation_treats_zero_and_180_degrees_as_equivalent() -> None:
    predicted = torch.tensor([[[[0.0]], [[1.0]]]])  # [sin(0), cos(0)]
    same_axis_angle = torch.tensor([[[math.pi]]])
    perpendicular_angle = torch.tensor([[[math.pi / 2.0]]])

    equivalent = axial_orientation_loss(
        predicted,
        same_axis_angle,
        target_is_angle=True,
    )
    perpendicular = axial_orientation_loss(
        predicted,
        perpendicular_angle,
        target_is_angle=True,
    )

    torch.testing.assert_close(equivalent, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(perpendicular, torch.tensor(2.0), atol=1e-6, rtol=0)


def test_soft_cldice_penalizes_a_broken_centerline() -> None:
    target = torch.zeros(1, 1, 15, 15)
    target[:, :, 7, 2:13] = 1.0
    complete = target.clone()
    broken = target.clone()
    broken[:, :, 7, 6:9] = 0.0

    complete_loss = soft_cldice_loss(
        complete,
        target,
        from_logits=False,
        iterations=5,
    )
    broken_loss = soft_cldice_loss(
        broken,
        target,
        from_logits=False,
        iterations=5,
    )

    assert complete_loss < broken_loss


def test_instance_bridge_loss_penalizes_only_corridors_between_different_rows() -> None:
    row_ids = torch.zeros(1, 15, 15, dtype=torch.int64)
    row_ids[:, 2:13, 4] = 1
    row_ids[:, 2:13, 10] = 2
    no_bridge = torch.full((1, 1, 15, 15), -8.0)
    no_bridge[:, :, 2:13, 4] = 8.0
    no_bridge[:, :, 2:13, 10] = 8.0
    no_bridge.requires_grad_(True)
    bridge = no_bridge.detach().clone().requires_grad_(True)
    with torch.no_grad():
        bridge[:, :, 7, 5:10] = 8.0

    weights = instance_corridor_weights(row_ids)
    baseline_loss = instance_bridge_loss(
        no_bridge,
        row_ids,
        corridor_weights=weights,
    )
    bridge_loss = instance_bridge_loss(
        bridge,
        row_ids,
        corridor_weights=weights,
    )

    assert weights[0, 0, 7, 7] > weights[0, 0, 7, 5] > 0
    assert baseline_loss < bridge_loss
    bridge_loss.backward()
    assert bridge.grad is not None
    assert bridge.grad[0, 0, 7, 7] > 0

    one_row = row_ids.clone()
    one_row[one_row == 2] = 1
    same_instance_loss = instance_bridge_loss(bridge, one_row)
    torch.testing.assert_close(same_instance_loss, torch.zeros_like(same_instance_loss))


def test_split_center_targets_ignore_occlusion_only_for_observed_loss() -> None:
    observed = torch.zeros(1, 1, 15, 15)
    observed[:, :, 7, 2:6] = 1.0
    observed[:, :, 7, 9:13] = 1.0
    complete = observed.clone()
    complete[:, :, 7, 6:9] = 1.0
    observed_logits = torch.where(observed.bool(), 8.0, -8.0).requires_grad_(True)
    broken_complete_logits = observed_logits.detach().clone().requires_grad_(True)
    filled_complete_logits = broken_complete_logits.detach().clone()
    filled_complete_logits[:, :, 7, 6:9] = 8.0
    filled_complete_logits.requires_grad_(True)
    targets = {
        "center_observed": observed,
        "center_complete": complete,
        "valid_mask": torch.ones(1, 15, 15, dtype=torch.bool),
    }
    criterion = RowGraphNetLoss(learned_weighting=False, cldice_iterations=5)

    broken_losses = criterion(
        {
            "center_observed_logits": observed_logits,
            "center_complete_logits": broken_complete_logits,
        },
        targets,
    )
    filled_losses = criterion(
        {
            "center_observed_logits": observed_logits,
            "center_complete_logits": filled_complete_logits,
        },
        targets,
    )

    torch.testing.assert_close(broken_losses["center"], filled_losses["center"])
    assert filled_losses["topology"] < broken_losses["topology"]
    broken_losses["total"].backward()
    assert observed_logits.grad is not None
    assert torch.count_nonzero(observed_logits.grad[:, :, 7, 6:9]) == 0
    assert broken_complete_logits.grad is not None


def test_composite_loss_is_finite_and_backpropagates_all_dense_heads() -> None:
    batch, height, width = 2, 10, 12
    orientation_raw = torch.randn(batch, 2, height, width, requires_grad=True)
    orientation = torch.nn.functional.normalize(orientation_raw, dim=1)
    predictions = {
        "band_logits": torch.randn(batch, 1, height, width, requires_grad=True),
        "center_logits": torch.randn(batch, 1, height, width, requires_grad=True),
        "orientation": orientation,
        "endpoints_logits": torch.randn(batch, 1, height, width, requires_grad=True),
        "distance": torch.rand(batch, 1, height, width, requires_grad=True),
        "uncertainty": torch.rand(batch, 1, height, width, requires_grad=True),
        "repair_logits": torch.randn(batch, 1, height, width, requires_grad=True),
        "edge_logits": torch.randn(4, requires_grad=True),
    }
    center = torch.zeros(batch, 1, height, width)
    center[:, :, 3, 2:10] = 1.0
    center[:, :, 7, 2:10] = 1.0
    row_ids = torch.zeros(batch, height, width, dtype=torch.int64)
    row_ids[:, 3, 2:10] = 1
    row_ids[:, 7, 2:10] = 2
    target_orientation = torch.zeros(batch, 2, height, width)
    target_orientation[:, 1] = 1.0
    targets = {
        "band": torch.randint(0, 2, (batch, 1, height, width)).float(),
        "center": center,
        "row_id_mask": row_ids,
        "orientation": target_orientation,
        "orientation_valid_mask": center.bool(),
        "endpoints": torch.zeros(batch, 1, height, width),
        "distance": torch.rand(batch, 1, height, width),
        "repair": torch.zeros(batch, 1, height, width),
        "edge_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "valid_mask": torch.ones(batch, height, width, dtype=torch.bool),
    }
    criterion = RowGraphNetLoss(cldice_iterations=3)

    losses = criterion(predictions, targets)

    expected = {
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
        "total",
    }
    assert set(losses) == expected
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    for name, value in predictions.items():
        if name == "orientation":
            continue
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name
    assert orientation_raw.grad is not None
    for parameter in criterion.log_variances.values():
        assert parameter.grad is not None


def test_composite_loss_excludes_empty_raster_tasks_from_learned_weighting() -> None:
    center_logits = torch.randn(1, 1, 6, 7, requires_grad=True)
    predictions = {
        "center_logits": center_logits,
        "center": torch.sigmoid(center_logits),
    }
    targets = {
        "center": torch.zeros_like(center_logits),
        "valid_mask": torch.zeros(1, 6, 7, dtype=torch.bool),
    }
    criterion = RowGraphNetLoss(cldice_iterations=2)

    losses = criterion(predictions, targets)

    torch.testing.assert_close(losses["center"], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(losses["topology"], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(losses["total"], torch.tensor(0.0), atol=1e-6, rtol=0)
    losses["total"].backward()
    assert center_logits.grad is not None
    assert torch.count_nonzero(center_logits.grad) == 0


def test_band_specific_validity_mask_disables_unavailable_supervision() -> None:
    band_logits = torch.randn(1, 1, 5, 7, requires_grad=True)
    criterion = RowGraphNetLoss(learned_weighting=False)
    losses = criterion(
        {"band_logits": band_logits},
        {
            "band": torch.ones_like(band_logits),
            "valid_mask": torch.ones_like(band_logits),
            "band_valid_mask": torch.zeros_like(band_logits),
        },
    )
    torch.testing.assert_close(losses["total"], torch.tensor(0.0), atol=1e-6, rtol=0)
    losses["total"].backward()
    assert torch.count_nonzero(band_logits.grad) == 0
