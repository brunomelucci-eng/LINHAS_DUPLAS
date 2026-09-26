from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import geopandas
import numpy
import rasterio
import scipy
import shapely
import torch
import torchvision


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "modelos" / "best_23_08_2026_12_imagens_run_001.pt"
CALIBRATION = MODEL.with_suffix(MODEL.suffix + ".threshold.json")
EXPECTED_SHA256 = {
    MODEL.name: "B2D87A429594BE8720137E0DE48831D98C897BA5E25B6A596CD4902922727716",
    CALIBRATION.name: "5092C1DC5FD6132EF94F7A824769B75C5EC2DF8B9FF00CA17FA7137781650B0A",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 x64 requerido; encontrado {sys.version.split()[0]}.")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Modelo ausente: {MODEL}")
    if not CALIBRATION.is_file():
        raise FileNotFoundError(f"Calibracao ausente: {CALIBRATION}")
    for path in (MODEL, CALIBRATION):
        actual = _sha256(path)
        if actual != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"Hash SHA256 invalido: {path.name}")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration.get("status") != "approved":
        raise RuntimeError("A calibracao do checkpoint nao esta aprovada.")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA disponivel: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Rasterio: {rasterio.__version__}")
    print(f"GeoPandas: {geopandas.__version__}")
    print(f"Shapely: {shapely.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Modelo: {MODEL.name}")
    print("Integridade SHA256: OK")
    print(f"Threshold calibrado: {calibration['center_threshold']}")
    print("INSTALACAO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
