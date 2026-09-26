import pytest

from src.data.spatial_split import allocate_group_counts, split_dataset_groups


@pytest.mark.parametrize(
    ("n_groups", "expected"),
    [(1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1)),
     (4, (2, 1, 1)), (5, (3, 1, 1)), (10, (6, 2, 2))],
)
def test_allocate_group_counts(n_groups, expected):
    assert allocate_group_counts(n_groups, 0.70, 0.15, 0.15) == expected


def test_split_does_not_leak_groups():
    tiles = [
        {"group_id": group_id, "bounds": (group_id, 0, group_id + 1, 1)}
        for group_id in range(5)
        for _ in range(2)
    ]
    train, val, test = split_dataset_groups(tiles, seed=42)
    train_ids = {tile["group_id"] for tile in train}
    val_ids = {tile["group_id"] for tile in val}
    test_ids = {tile["group_id"] for tile in test}
    assert (len(train_ids), len(val_ids), len(test_ids)) == (3, 1, 1)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
