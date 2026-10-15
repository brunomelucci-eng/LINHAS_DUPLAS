"""Deterministic spatial splits whose indivisible unit is ``group_id``."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from rowgraphnet.data.manifest import ALLOWED_SPLITS, ManifestError, SceneManifest, SplitName


class SplitLeakageError(ManifestError):
    """Raised when one property/field group occurs in multiple splits."""


@dataclass(frozen=True, slots=True)
class SplitFractions:
    """Relative group counts for the four allowed dataset partitions."""

    train: float = 0.7
    validation: float = 0.15
    calibration: float = 0.05
    test: float = 0.1

    def __post_init__(self) -> None:
        values = self.as_mapping()
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("split fractions must be finite and non-negative")
        if sum(values.values()) <= 0.0:
            raise ValueError("at least one split fraction must be positive")

    def as_mapping(self) -> dict[SplitName, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "calibration": self.calibration,
            "test": self.test,
        }


def validate_group_isolation(scenes: Iterable[SceneManifest]) -> dict[str, SplitName]:
    """Return group assignments, rejecting leakage across dataset partitions."""

    assignments: dict[str, SplitName] = {}
    for scene in scenes:
        previous = assignments.get(scene.group_id)
        if previous is not None and previous != scene.split:
            raise SplitLeakageError(
                f"group_id {scene.group_id!r} leaks across splits {previous!r} and "
                f"{scene.split!r} (scene_id={scene.scene_id!r})"
            )
        assignments[scene.group_id] = scene.split
    return assignments


def _stable_group_order(group_ids: Sequence[str], seed: int | str) -> list[str]:
    seed_bytes = str(seed).encode("utf-8")

    def key(group_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(seed_bytes + b"\0" + group_id.encode("utf-8")).digest()
        return digest, group_id

    return sorted(group_ids, key=key)


def _allocate_counts(total: int, fractions: Mapping[SplitName, float]) -> dict[SplitName, int]:
    order: tuple[SplitName, ...] = ("train", "validation", "calibration", "test")
    fraction_sum = sum(fractions.values())
    targets = {name: total * fractions[name] / fraction_sum for name in order}
    counts = {name: math.floor(targets[name]) for name in order}
    remaining = total - sum(counts.values())
    priority = sorted(order, key=lambda name: (-(targets[name] - counts[name]), order.index(name)))
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def generate_group_split_map(
    group_ids: Iterable[str],
    *,
    fractions: SplitFractions | Mapping[str, float] | None = None,
    seed: int | str = 0,
) -> dict[str, SplitName]:
    """Assign whole groups with deterministic hashing and balanced counts.

    Tile identifiers are intentionally not accepted. Callers provide property,
    field, or acquisition-group identifiers; every scene sharing an identifier
    receives exactly one partition.
    """

    normalized: list[str] = []
    for group_id in group_ids:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_ids must contain non-empty strings")
        normalized.append(group_id.strip())
    unique = sorted(set(normalized))
    if not unique:
        raise ValueError("at least one group_id is required")

    if fractions is None:
        fraction_map = SplitFractions().as_mapping()
    elif isinstance(fractions, SplitFractions):
        fraction_map = fractions.as_mapping()
    elif isinstance(fractions, Mapping):
        unknown = set(fractions) - ALLOWED_SPLITS
        missing = ALLOWED_SPLITS - set(fractions)
        if unknown or missing:
            raise ValueError(
                f"split fractions must define exactly {sorted(ALLOWED_SPLITS)}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        try:
            fraction_map = {name: float(fractions[name]) for name in ALLOWED_SPLITS}
        except (TypeError, ValueError) as exc:
            raise ValueError("split fractions must be numeric") from exc
        fraction_map = SplitFractions(**fraction_map).as_mapping()
    else:
        raise TypeError("fractions must be SplitFractions or a mapping")

    counts = _allocate_counts(len(unique), fraction_map)
    ordered_groups = _stable_group_order(unique, seed)
    assignments: dict[str, SplitName] = {}
    offset = 0
    for split in ("train", "validation", "calibration", "test"):
        for group_id in ordered_groups[offset : offset + counts[split]]:
            assignments[group_id] = split
        offset += counts[split]
    return assignments


def assign_group_splits(
    scenes: Iterable[SceneManifest],
    *,
    fractions: SplitFractions | Mapping[str, float] | None = None,
    seed: int | str = 0,
) -> tuple[SceneManifest, ...]:
    """Return copies with deterministic splits, preserving scene input order."""

    materialized = tuple(scenes)
    if not materialized:
        raise ValueError("at least one scene is required")
    scene_counts = Counter(scene.scene_id for scene in materialized)
    duplicates = sorted(scene_id for scene_id, count in scene_counts.items() if count > 1)
    if duplicates:
        raise ManifestError(f"duplicate scene_id values: {duplicates}")
    assignments = generate_group_split_map(
        (scene.group_id for scene in materialized),
        fractions=fractions,
        seed=seed,
    )
    assigned = tuple(replace(scene, split=assignments[scene.group_id]) for scene in materialized)
    validate_group_isolation(assigned)
    return assigned
