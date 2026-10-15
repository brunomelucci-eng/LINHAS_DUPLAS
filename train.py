"""Backward-compatible launcher for the packaged RowGraphNet training runner."""

import sys
from pathlib import Path

_SOURCE = str(Path(__file__).resolve().parent / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from rowgraphnet.training.runner import main, train_rowgraphnet  # noqa: E402

__all__ = ["main", "train_rowgraphnet"]


if __name__ == "__main__":
    main()
