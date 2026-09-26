"""Backward-compatible entry point for network cleanup."""

from typing import List, Tuple

from .network_cleanup import cleanup_network


def resolve_junctions(
    branches: List[List[Tuple[int, int]]],
    config: dict,
    gsd: float = 1.0,
) -> List[List[Tuple[int, int]]]:
    return cleanup_network(branches, config, gsd=gsd)
