import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from typing import List, Dict, Any, Tuple
import rasterio
import logging
import random

logger = logging.getLogger(__name__)

class TileGenerator:
    def __init__(
        self,
        tile_size_px: int = 512,
        overlap_px: int = 128,
        min_valid_fraction: float = 0.60,
        include_empty_tiles_ratio: float = 0.15,
        negative_sampling_mode: str = "explicit_zones",
        negative_zone_min_fraction: float = 0.80,
        min_negative_line_distance_m: float = 0.0,
    ):
        self.tile_size_px = tile_size_px
        self.overlap_px = overlap_px
        self.min_valid_fraction = min_valid_fraction
        self.include_empty_tiles_ratio = include_empty_tiles_ratio
        if negative_sampling_mode not in {"explicit_zones", "unlabeled_empty"}:
            raise ValueError("negative_sampling_mode must be explicit_zones or unlabeled_empty")
        self.negative_sampling_mode = negative_sampling_mode
        self.negative_zone_min_fraction = float(negative_zone_min_fraction)
        self.min_negative_line_distance_m = float(min_negative_line_distance_m)

    def generate_tiles(
        self,
        raster_width: int,
        raster_height: int,
        raster_transform: rasterio.Affine,
        roi_gdf: gpd.GeoDataFrame,
        group_id: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Generate tile bounds (col_off, row_off, width, height) and metadata.
        """
        tiles = []
        roi_geom = roi_gdf.unary_union
        
        step_px = self.tile_size_px - self.overlap_px
        if step_px <= 0:
            raise ValueError("Overlap must be strictly smaller than tile size.")
            
        for y in range(0, raster_height, step_px):
            row_off = y
            h = self.tile_size_px
            if row_off + h > raster_height:
                row_off = max(0, raster_height - h)
                
            for x in range(0, raster_width, step_px):
                col_off = x
                w = self.tile_size_px
                if col_off + w > raster_width:
                    col_off = max(0, raster_width - w)
                    
                # Corner points in world coordinates
                xs, ys = rasterio.transform.xy(
                    raster_transform, 
                    [row_off, row_off+h, row_off+h, row_off], 
                    [col_off, col_off, col_off+w, col_off+w]
                )
                tile_poly = Polygon(zip(xs, ys))
                
                if tile_poly.intersects(roi_geom):
                    intersection_area = tile_poly.intersection(roi_geom).area
                    tile_area = tile_poly.area
                    valid_fraction = intersection_area / tile_area
                else:
                    valid_fraction = 0.0
                    
                if valid_fraction >= self.min_valid_fraction:
                    minx, miny, maxx, maxy = tile_poly.bounds
                    tiles.append({
                        'col_off': col_off,
                        'row_off': row_off,
                        'width': w,
                        'height': h,
                        'valid_fraction': valid_fraction,
                        'bounds': (minx, miny, maxx, maxy),
                        'geom': tile_poly,
                        'group_id': group_id
                    })
                    
        # Remove duplicate tiles if step size caused overlaps at boundaries to snap to same offsets
        unique_tiles = []
        seen = set()
        for t in tiles:
            key = (t['col_off'], t['row_off'], t['width'], t['height'])
            if key not in seen:
                seen.add(key)
                unique_tiles.append(t)
                
        return unique_tiles

    def filter_tiles(
        self,
        tiles: List[Dict[str, Any]],
        lines_gdf: gpd.GeoDataFrame,
        hard_negative_zones_gdf: gpd.GeoDataFrame = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Filter tiles based on overlap with reference lines and include_empty_tiles_ratio.
        """
        if not tiles:
            return []
            
        random_gen = random.Random(seed)
        lines_union = lines_gdf.geometry.union_all()
        negative_union = None
        if hard_negative_zones_gdf is not None and not hard_negative_zones_gdf.empty:
            negative_union = hard_negative_zones_gdf.geometry.union_all()
        
        positive_tiles = []
        empty_tiles = []
        
        for tile in tiles:
            tile_geom = tile['geom']
            if tile_geom.intersects(lines_union):
                tile['is_empty'] = False
                tile['sampling_class'] = 'positive'
                positive_tiles.append(tile)
            else:
                eligible = self.negative_sampling_mode == "unlabeled_empty"
                if self.negative_sampling_mode == "explicit_zones" and negative_union is not None:
                    coverage = tile_geom.intersection(negative_union).area / max(tile_geom.area, 1e-9)
                    eligible = coverage >= self.negative_zone_min_fraction
                if eligible and tile_geom.distance(lines_union) >= self.min_negative_line_distance_m:
                    tile['is_empty'] = True
                    tile['sampling_class'] = 'hard_negative'
                    empty_tiles.append(tile)
                
        num_empty_to_keep = int(len(positive_tiles) * self.include_empty_tiles_ratio)
        if num_empty_to_keep >= len(empty_tiles):
            kept_empty = empty_tiles
        else:
            kept_empty = random_gen.sample(empty_tiles, num_empty_to_keep)
            
        logger.info(
            f"Tiling: total tiles generated={len(tiles)}, "
            f"positive_tiles={len(positive_tiles)}, "
            f"hard_negative_tiles_kept={len(kept_empty)} (eligible={len(empty_tiles)}, "
            f"mode={self.negative_sampling_mode})"
        )
        
        filtered = positive_tiles + kept_empty
        # Shuffle to mix positive and empty tiles
        random_gen.shuffle(filtered)
        return filtered
