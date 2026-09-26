import json
from datetime import datetime

from src.training.trainer import _dataset_image_count, _unique_checkpoint_paths


def test_checkpoint_name_contains_date_image_count_and_run_number(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    now = datetime(2026, 8, 22)

    best, last = _unique_checkpoint_paths(str(checkpoint_dir), 12, now=now)

    assert best.endswith("best_22_08_2026_12_imagens_run_001.pt")
    assert last.endswith("last_22_08_2026_12_imagens_run_001.pt")

    (checkpoint_dir / "best_22_08_2026_12_imagens_run_001.pt").touch()
    best, last = _unique_checkpoint_paths(str(checkpoint_dir), 12, now=now)
    assert best.endswith("best_22_08_2026_12_imagens_run_002.pt")
    assert last.endswith("last_22_08_2026_12_imagens_run_002.pt")


def test_dataset_image_count_prefers_prepared_manifest(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"inputs": [{"input_name": "a"}, {"input_name": "b"}]}),
        encoding="utf-8",
    )

    assert _dataset_image_count({"inputs": [{}, {}, {}]}, str(tmp_path)) == 2
