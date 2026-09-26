"""Leak-free spatial grouping and split allocation."""

import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def allocate_group_counts(
    n_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    """Allocate whole spatial groups while preserving minimum viable splits."""
    if n_groups <= 0:
        return 0, 0, 0
    if n_groups == 1:
        return 1, 0, 0
    if n_groups == 2:
        return 1, 1, 0

    n_val = max(1, int(round(n_groups * val_ratio)))
    n_test = max(1, int(round(n_groups * test_ratio)))
    n_train = n_groups - n_val - n_test
    while n_train < 1:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Unable to reserve a train, validation, and test group.")
        n_train = n_groups - n_val - n_test
    return n_train, n_val, n_test


def _cluster_tiles(tiles: List[Dict[str, Any]], seed: int) -> None:
    if not tiles:
        return
    n_clusters = min(len(tiles), max(1, min(max(5, len(tiles) // 10), 30)))
    if n_clusters == 1:
        for tile in tiles:
            tile["group_id"] = 0
        return
    centroids = np.array([
        [(tile["bounds"][0] + tile["bounds"][2]) / 2.0,
         (tile["bounds"][1] + tile["bounds"][3]) / 2.0]
        for tile in tiles
    ])
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(centroids)
    for tile, label in zip(tiles, labels):
        tile["group_id"] = int(label)


def split_dataset_groups(
    tiles: List[Dict[str, Any]],
    strategy: str = "spatial_group",
    group_field: str = "talhao_id",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split complete spatial groups without leaking a group between splits.

    A source with one group is train-only; two groups are train/validation;
    sources with three or more groups always reserve train/validation/test.
    """
    if not tiles:
        return [], [], []
    if not (strategy == "spatial_group" and group_field and all(t.get("group_id") is not None for t in tiles)):
        logger.info("Creating spatial clusters because explicit group IDs are unavailable.")
        _cluster_tiles(tiles, seed)

    groups = list({tile["group_id"] for tile in tiles})
    random.Random(seed).shuffle(groups)
    n_train, n_val, n_test = allocate_group_counts(
        len(groups), train_fraction, val_fraction, test_fraction,
    )
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:n_train + n_val + n_test])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Spatial group allocation overlaps between splits.")

    train_tiles = [tile for tile in tiles if tile["group_id"] in train_groups]
    val_tiles = [tile for tile in tiles if tile["group_id"] in val_groups]
    test_tiles = [tile for tile in tiles if tile["group_id"] in test_groups]
    logger.info(
        "Split report: Train=%d tiles (%d groups), Val=%d tiles (%d groups), Test=%d tiles (%d groups)",
        len(train_tiles), len(train_groups), len(val_tiles), len(val_groups), len(test_tiles), len(test_groups),
    )
    return train_tiles, val_tiles, test_tiles
