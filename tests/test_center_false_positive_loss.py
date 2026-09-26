import torch

from src.losses import CombinedLoss


def _targets():
    zeros = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    return {
        "row_mask": zeros.clone(),
        "center_mask": zeros.clone(),
        "orientation_sin": zeros.clone(),
        "orientation_cos": zeros.clone(),
        "valid_mask": torch.ones_like(zeros),
    }


def test_false_positive_term_penalizes_center_probability_on_background():
    loss = CombinedLoss({"loss": {"center_false_positive_weight": 1.0}})
    low = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    high = low.clone()
    low[:, 1] = -5.0
    high[:, 1] = 5.0

    low_term = loss(low, _targets())["center_false_positive"]
    high_term = loss(high, _targets())["center_false_positive"]

    assert high_term > low_term
