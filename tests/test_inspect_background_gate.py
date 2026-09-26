import json
import sys

import numpy as np
import yaml

from scripts.inspect_data import main


def _write_tile(path):
    center = np.zeros((8, 8), dtype=np.uint8)
    center[3, :] = 1
    np.savez_compressed(
        path,
        image=np.zeros((3, 8, 8), dtype=np.float32),
        row_mask=center,
        center_mask=center,
        orientation_sin=np.zeros((8, 8), dtype=np.float32),
        orientation_cos=np.zeros((8, 8), dtype=np.float32),
        valid_mask=np.ones((8, 8), dtype=np.uint8),
        input_name="dense_field",
        group_id=path.parent.name,
        sampling_class="positive",
    )


def test_dense_positive_tiles_are_approved_when_background_pixels_are_sufficient(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        directory = dataset / split
        directory.mkdir(parents=True)
        _write_tile(directory / "tile.npz")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_quality": {
                    "min_hard_negative_tiles": 0,
                    "min_background_pixel_fraction": 0.70,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_data.py",
            "--dataset-dir",
            str(dataset),
            "--output",
            str(output),
            "--config",
            str(config_path),
        ],
    )

    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "approved"
    assert report["hard_negative_tiles"] == 0
    assert report["background_pixel_fraction"] == 0.875
