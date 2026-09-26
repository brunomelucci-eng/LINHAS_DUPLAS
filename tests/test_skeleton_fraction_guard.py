import numpy as np
import pytest

from src.postprocessing.skeleton_graph import skeleton_to_graph_and_branches


def test_relative_skeleton_safety_limit_rejects_saturated_mask():
    skeleton = np.ones((20, 20), dtype=np.uint8)
    config = {
        "postprocessing": {
            "graph": {
                "backend": "numpy",
                "max_skeleton_fraction": 0.08,
                "max_total_skeleton_pixels": 1_000_000,
            }
        }
    }
    with pytest.raises(RuntimeError, match="adaptive safety limit"):
        skeleton_to_graph_and_branches(skeleton, config)
