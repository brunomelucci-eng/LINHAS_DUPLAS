import rasterio
import numpy as np
import os
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class RasterReader:
    def __init__(self, filepath: str, layer: Optional[str] = None):
        self.filepath = filepath
        self.layer = layer
        self.src = None
        self._open()

    def _open(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Raster file not found: {self.filepath}")
        
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == '.gpkg':
            # Check subdatasets
            with rasterio.open(self.filepath) as tmp_src:
                subdatasets = tmp_src.subdatasets
            
            if subdatasets:
                if self.layer is None:
                    if len(subdatasets) == 1:
                        dataset_path = subdatasets[0]
                    else:
                        raise ValueError(
                            f"GPKG raster contains multiple layers: {subdatasets}. "
                            "Please specify the 'orthomosaic_layer' config parameter."
                        )
                else:
                    matching = [s for s in subdatasets if self.layer in s]
                    if not matching:
                        raise ValueError(f"Layer '{self.layer}' not found in GPKG subdatasets: {subdatasets}")
                    dataset_path = matching[0]
                
                logger.info(f"Opening GPKG subdataset: {dataset_path}")
                self.src = rasterio.open(dataset_path)
            else:
                self.src = rasterio.open(self.filepath)
        else:
            self.src = rasterio.open(self.filepath)

    @property
    def crs(self):
        return self.src.crs

    @property
    def transform(self):
        return self.src.transform

    @property
    def width(self):
        return self.src.width

    @property
    def height(self):
        return self.src.height

    @property
    def bounds(self):
        return self.src.bounds

    @property
    def count(self):
        return self.src.count

    @property
    def nodata(self):
        return self.src.nodata

    def read_bands(
        self, 
        bands: List[int], 
        window: Optional[rasterio.windows.Window] = None,
        out_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Read specific bands and return a numpy array of shape (C, H, W).
        """
        # Validate bands
        for b in bands:
            if b < 1 or b > self.count:
                raise ValueError(f"Band index {b} is out of bounds (1 to {self.count}).")
        
        if out_shape is not None:
            data = self.src.read(bands, window=window, out_shape=(len(bands), out_shape[0], out_shape[1]))
        else:
            data = self.src.read(bands, window=window)
        return data

    def get_gsd(self) -> float:
        """
        Get Ground Sampling Distance (resolution in meters per pixel).
        """
        if self.crs and self.crs.is_geographic:
            logger.warning("Calculando GSD para CRS geográfico. O valor retornado estará em graus/pixel, não em metros.")
        t = self.transform
        return (abs(t[0]) + abs(t[4])) / 2.0

    def read_normalized(
        self, 
        bands: List[int], 
        window: Optional[rasterio.windows.Window] = None,
        out_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Read bands and normalize to [0, 1] float32.
        """
        data = self.read_bands(bands, window=window, out_shape=out_shape).astype(np.float32)
        nodata_val = self.nodata
        mask = None
        if nodata_val is not None:
            # Handle float comparison safety
            if np.isnan(nodata_val):
                mask = np.isnan(data)
            else:
                mask = (data == nodata_val)
        
        # Global consistent scaling based on native dtype to prevent sliding-window seam artifacts
        native_dtype = str(self.src.dtypes[0]).lower() if (self.src and self.src.dtypes) else 'uint8'
        if 'uint8' in native_dtype:
            data = data / 255.0
        elif 'uint16' in native_dtype:
            data = data / 65535.0
        else:
            max_val = float(np.nanmax(data)) if data.size > 0 else 1.0
            if max_val > 1.0:
                data = data / 255.0

        data = np.clip(data, 0.0, 1.0)
        if mask is not None:
            data[mask] = 0.0
            
        return data

    def close(self):
        if self.src:
            self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
