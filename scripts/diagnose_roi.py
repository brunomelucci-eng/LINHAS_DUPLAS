"""Run the production pipeline on a small ROI and produce diagnostic metrics."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.diagnostics import build_roi_diagnostic_report, write_diagnostic_reports


ARTEFACTS = (
    "probability_center.tif",
    "mosaic_weights.tif",
    "binary_mask_raw.tif",
    "binary_mask_cleaned.tif",
    "skeleton.tif",
    "tile_grid.gpkg",
)


def _existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one small inference ROI.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--orthomosaic", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--output", required=True, help="Diagnostic output directory.")
    args = parser.parse_args()

    config_path = _existing_file(args.config, "Config")
    checkpoint = _existing_file(args.checkpoint, "Checkpoint")
    orthomosaic = _existing_file(args.orthomosaic, "Orthomosaic")
    roi = _existing_file(args.roi, "ROI")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))
    config.setdefault("project", {})["output_dir"] = str(output)
    config.setdefault("input", {})["orthomosaic_layer"] = None
    config.setdefault("input", {})["roi_layer"] = None
    config["debug"] = {
        "enabled": True,
        "save_probability": True,
        "save_weights": True,
        "save_masks": True,
        "save_skeleton": True,
        "save_tile_grid": True,
    }
    generated_config = output / "diagnostic_config.yaml"
    generated_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    final_gpkg = output / "linhas_finais.gpkg"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "predict.py"),
        "--config", str(generated_config),
        "--checkpoint", str(checkpoint),
        "--orthomosaic", str(orthomosaic),
        "--roi", str(roi),
        "--output", str(final_gpkg),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    runs = sorted((output / "runs").glob("*"), key=lambda path: path.stat().st_mtime)
    if not runs:
        raise RuntimeError("Prediction completed without creating an isolated run directory.")
    run_dir = runs[-1]
    for name in ARTEFACTS:
        source = run_dir / name
        if not source.is_file():
            raise RuntimeError(f"Mandatory diagnostic artefact was not created: {source}")
        shutil.copy2(source, output / name)

    report = build_roi_diagnostic_report(
        output / "probability_center.tif",
        output / "binary_mask_cleaned.tif",
        output / "skeleton.tif",
        config,
        mosaic_weights_path=output / "mosaic_weights.tif",
        checkpoint=str(checkpoint),
        orthomosaic=str(orthomosaic),
        roi=str(roi),
    )
    write_diagnostic_reports(report, output)
    print(f"Diagnostic completed: {output}")


if __name__ == "__main__":
    main()
