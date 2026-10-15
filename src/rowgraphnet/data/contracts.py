"""Validation contracts for georeferenced rasters and polygonal ROIs.

All geometric measurements downstream of this module use a projected CRS in
metres.  Geographic rasters remain in their native grid for reading, while a
local UTM/UPS CRS is selected from the actual analysis footprint for metric
operations.  Pixel size is measured by transforming the affine basis vectors
at the ROI, so rotated rasters and geographic grids are handled correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.errors import RasterioIOError
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union


class DataContractError(ValueError):
    """Raised when an input cannot satisfy the geospatial data contract."""


@dataclass(frozen=True, slots=True)
class PixelSize:
    """Physical lengths of the raster column and row basis vectors."""

    x_m: float
    y_m: float

    @property
    def mean_m(self) -> float:
        """Geometric mean, suitable for converting isotropic target widths."""

        return math.sqrt(self.x_m * self.y_m)


@dataclass(frozen=True, slots=True)
class RasterContract:
    """Validated, lightweight raster metadata (no raster pixels are retained)."""

    path: Path
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: CRS
    metric_crs: CRS
    transform: Affine
    footprint: BaseGeometry
    bands: tuple[int, ...]
    nodata_values: tuple[float | None, ...]
    gsd: PixelSize


@dataclass(frozen=True, slots=True)
class ValidatedScene:
    """Raster contract plus the ROI represented in all required CRSs."""

    raster: RasterContract
    roi_in_raster_crs: BaseGeometry
    analysis_roi_in_raster_crs: BaseGeometry
    roi_in_metric_crs: BaseGeometry
    roi_was_clipped: bool
    invalid_mask_path: Path | None = None


def _as_crs(value: Any, *, name: str) -> CRS:
    if value is None:
        raise DataContractError(f"{name} must define a CRS")
    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:  # pyproj raises several input-specific subclasses
        raise DataContractError(f"{name} has an invalid CRS: {value!r}") from exc
    if not crs.is_geographic and not crs.is_projected:
        raise DataContractError(f"{name} CRS must be geographic or projected")
    return crs


def is_metric_crs(crs: Any) -> bool:
    """Return whether both horizontal axes of a projected CRS use metres."""

    parsed = _as_crs(crs, name="CRS")
    if not parsed.is_projected or len(parsed.axis_info) < 2:
        return False
    axes = parsed.axis_info[:2]
    return all(
        math.isfinite(axis.unit_conversion_factor)
        and math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and axis.unit_name.lower() in {"metre", "meter", "metres", "meters"}
        for axis in axes
    )


def transform_geometry(geometry: BaseGeometry, source_crs: Any, target_crs: Any) -> BaseGeometry:
    """Transform a geometry with explicit XY axis order and finite-output checks."""

    if geometry is None or geometry.is_empty:
        raise DataContractError("geometry must not be empty")
    source = _as_crs(source_crs, name="source")
    target = _as_crs(target_crs, name="target")
    if source == target:
        result = geometry
    else:
        transformer = Transformer.from_crs(source, target, always_xy=True)
        try:
            result = shapely_transform(transformer.transform, geometry)
        except Exception as exc:
            raise DataContractError(
                f"could not transform geometry from {source} to {target}"
            ) from exc
    if result.is_empty or not np.isfinite(np.asarray(result.bounds, dtype=np.float64)).all():
        raise DataContractError("geometry transformation produced empty or non-finite coordinates")
    return result


def _utm_zone(longitude: float, latitude: float) -> int:
    """Select the WGS84 UTM zone, including the Norway/Svalbard exceptions."""

    longitude = min(max(longitude, -180.0), 180.0)
    zone = min(60, max(1, math.floor((longitude + 180.0) / 6.0) + 1))
    if 56.0 <= latitude < 64.0 and 3.0 <= longitude < 12.0:
        return 32
    if 72.0 <= latitude < 84.0:
        if 0.0 <= longitude < 9.0:
            return 31
        if 9.0 <= longitude < 21.0:
            return 33
        if 21.0 <= longitude < 33.0:
            return 35
        if 33.0 <= longitude < 42.0:
            return 37
    return zone


def choose_local_metric_crs(source_crs: Any, area: BaseGeometry) -> CRS:
    """Choose a metre-based CRS local to ``area``.

    Existing projected metre CRSs are retained, except global Web Mercator
    variants whose nominal metre is substantially distorted away from the
    equator.  Geographic/non-metric inputs use WGS84 UTM, or UPS in the polar
    regions.
    """

    source = _as_crs(source_crs, name="source")
    if is_metric_crs(source) and source.to_epsg() not in {3395, 3857}:
        return source

    area_wgs84 = transform_geometry(area, source, CRS.from_epsg(4326))
    point = area_wgs84.representative_point()
    longitude, latitude = float(point.x), float(point.y)
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise DataContractError("ROI does not map to a valid longitude/latitude")
    if latitude >= 84.0:
        metric = CRS.from_epsg(32661)  # WGS84 / UPS North
    elif latitude <= -80.0:
        metric = CRS.from_epsg(32761)  # WGS84 / UPS South
    else:
        zone = _utm_zone(longitude, latitude)
        metric = CRS.from_epsg((32600 if latitude >= 0.0 else 32700) + zone)

    # Fail here, not deep in post-processing, when projection is unsuitable.
    transformed = transform_geometry(area, source, metric)
    if not transformed.is_valid:
        from shapely import make_valid

        transformed = make_valid(transformed)
        if not transformed.is_valid:
            transformed = transformed.buffer(0)
    if not transformed.is_valid:
        raise DataContractError("ROI became invalid in the selected local metric CRS")
    return metric


def pixel_size_in_metres(
    transform: Affine,
    source_crs: Any,
    metric_crs: Any,
    location_xy: tuple[float, float],
) -> PixelSize:
    """Measure affine column/row vectors at a representative map location."""

    values = np.asarray(tuple(transform)[:6], dtype=np.float64)
    if not np.isfinite(values).all() or math.isclose(transform.determinant, 0.0, abs_tol=1e-18):
        raise DataContractError("raster affine transform is singular or non-finite")
    source = _as_crs(source_crs, name="raster")
    target = _as_crs(metric_crs, name="metric")
    transformer = Transformer.from_crs(source, target, always_xy=True)
    x0, y0 = location_xy
    source_points = (
        (x0, y0),
        (x0 + transform.a, y0 + transform.d),
        (x0 + transform.b, y0 + transform.e),
    )
    try:
        metric_points = [transformer.transform(x, y) for x, y in source_points]
    except Exception as exc:
        raise DataContractError("could not calculate raster GSD in metres") from exc
    if not np.isfinite(np.asarray(metric_points, dtype=np.float64)).all():
        raise DataContractError("raster GSD transformation produced non-finite coordinates")
    origin, column, row = (np.asarray(point, dtype=np.float64) for point in metric_points)
    x_m = float(np.linalg.norm(column - origin))
    y_m = float(np.linalg.norm(row - origin))
    if not (math.isfinite(x_m) and math.isfinite(y_m) and x_m > 0.0 and y_m > 0.0):
        raise DataContractError("raster GSD must be finite and positive")
    return PixelSize(x_m=x_m, y_m=y_m)


def validate_band_indexes(bands: Any, raster_count: int) -> tuple[int, ...]:
    """Validate explicit one-based Rasterio band indexes."""

    if raster_count < 1:
        raise DataContractError("raster must contain at least one band")
    try:
        selected = tuple(bands)
    except TypeError as exc:
        raise DataContractError("bands must be a non-empty sequence of one-based indexes") from exc
    if not selected:
        raise DataContractError("at least one raster band must be selected")
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in selected
    ):
        raise DataContractError("band indexes must be integers")
    normalized = tuple(int(index) for index in selected)
    if len(set(normalized)) != len(normalized):
        raise DataContractError("band indexes must be unique")
    invalid = [index for index in normalized if index < 1 or index > raster_count]
    if invalid:
        raise DataContractError(
            f"band indexes {invalid} are outside the raster range 1..{raster_count}"
        )
    return normalized


def _raster_footprint(transform: Affine, width: int, height: int) -> Polygon:
    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (width, height),
        transform * (0, height),
    ]
    footprint = Polygon(corners)
    if footprint.is_empty or not footprint.is_valid or footprint.area <= 0.0:
        raise DataContractError("raster transform does not define a valid footprint")
    return footprint


def validate_raster(
    raster_path: str | PathLike[str],
    *,
    bands: tuple[int, ...] | list[int] = (1, 2, 3),
    metric_reference: BaseGeometry | None = None,
) -> RasterContract:
    """Validate raster metadata without loading the full orthomosaic."""

    path = Path(raster_path).expanduser().resolve()
    try:
        with rasterio.open(path) as dataset:
            if dataset.width < 1 or dataset.height < 1:
                raise DataContractError("raster width and height must be positive")
            crs = _as_crs(dataset.crs, name="raster")
            transform = dataset.transform
            footprint = _raster_footprint(transform, dataset.width, dataset.height)
            selected = validate_band_indexes(bands, dataset.count)
            reference = metric_reference or footprint
            if not reference.intersects(footprint):
                raise DataContractError("metric reference does not intersect the raster")
            reference = reference.intersection(footprint)
            metric_crs = choose_local_metric_crs(crs, reference)
            location = reference.representative_point()
            gsd = pixel_size_in_metres(
                transform,
                crs,
                metric_crs,
                (float(location.x), float(location.y)),
            )
            return RasterContract(
                path=path,
                width=dataset.width,
                height=dataset.height,
                count=dataset.count,
                dtypes=tuple(dataset.dtypes[index - 1] for index in selected),
                crs=crs,
                metric_crs=metric_crs,
                transform=transform,
                footprint=footprint,
                bands=selected,
                nodata_values=tuple(dataset.nodatavals[index - 1] for index in selected),
                gsd=gsd,
            )
    except DataContractError:
        raise
    except (RasterioIOError, OSError) as exc:
        raise DataContractError(f"could not open raster: {path}") from exc


def _load_geometry(
    value: str | PathLike[str] | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
    supplied_crs: Any | None,
    *,
    name: str,
) -> tuple[BaseGeometry, CRS]:
    if isinstance(value, BaseGeometry):
        geometry = value
        crs = _as_crs(supplied_crs, name=name)
    else:
        if isinstance(value, (str, PathLike)):
            try:
                frame = gpd.read_file(value)
            except Exception as exc:
                raise DataContractError(f"could not read {name} vector data: {value}") from exc
        elif isinstance(value, gpd.GeoSeries):
            frame = gpd.GeoDataFrame(geometry=value, crs=value.crs)
        elif isinstance(value, gpd.GeoDataFrame):
            frame = value
        else:
            raise DataContractError(f"unsupported {name} input type: {type(value).__name__}")
        if frame.empty:
            raise DataContractError(f"{name} contains no features")
        declared_crs = _as_crs(frame.crs, name=name)
        if supplied_crs is not None:
            explicit_crs = _as_crs(supplied_crs, name=name)
            if explicit_crs != declared_crs:
                raise DataContractError(
                    f"{name} CRS argument conflicts with the CRS declared by the vector data"
                )
        crs = declared_crs
        geometries = [geometry for geometry in frame.geometry if geometry is not None]
        if not geometries:
            raise DataContractError(f"{name} contains no geometries")
        geometry = unary_union(geometries)

    if geometry.is_empty:
        raise DataContractError(f"{name} geometry must not be empty")
    if not geometry.is_valid:
        from shapely import make_valid

        geometry = make_valid(geometry)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
    if not geometry.is_valid:
        raise DataContractError(f"{name} geometry is invalid")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} and hasattr(geometry, "geoms"):
        from shapely.geometry import MultiPolygon

        poly_parts = [g for g in geometry.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        flat = []
        for p in poly_parts:
            if p.geom_type == "MultiPolygon":
                flat.extend(list(p.geoms))
            else:
                flat.append(p)
        if flat:
            geometry = flat[0] if len(flat) == 1 else MultiPolygon(flat)
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise DataContractError(f"{name} must contain polygonal geometry")
    if not np.isfinite(np.asarray(geometry.bounds, dtype=np.float64)).all():
        raise DataContractError(f"{name} geometry contains non-finite coordinates")
    return geometry, crs


def _validate_mask(mask_path: str | PathLike[str], raster: RasterContract) -> Path:
    path = Path(mask_path).expanduser().resolve()
    try:
        with rasterio.open(path) as mask:
            mask_crs = _as_crs(mask.crs, name="invalid mask")
            if mask.width != raster.width or mask.height != raster.height:
                raise DataContractError("invalid mask dimensions do not match the raster")
            if mask_crs != raster.crs:
                raise DataContractError("invalid mask CRS does not match the raster")
            if not mask.transform.almost_equals(raster.transform):
                raise DataContractError("invalid mask transform does not match the raster")
            if mask.count < 1:
                raise DataContractError("invalid mask has no bands")
    except DataContractError:
        raise
    except (RasterioIOError, OSError) as exc:
        raise DataContractError(f"could not open invalid mask: {path}") from exc
    return path


def validate_scene(
    raster_path: str | PathLike[str],
    roi: str | PathLike[str] | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
    *,
    roi_crs: Any | None = None,
    bands: tuple[int, ...] | list[int] = (1, 2, 3),
    invalid_mask_path: str | PathLike[str] | None = None,
) -> ValidatedScene:
    """Validate a raster/ROI pair and establish its local metric workspace."""

    raster = validate_raster(raster_path, bands=bands)
    roi_geometry, input_crs = _load_geometry(roi, roi_crs, name="ROI")
    roi_in_raster = transform_geometry(roi_geometry, input_crs, raster.crs)
    analysis_roi = roi_in_raster.intersection(raster.footprint)
    if analysis_roi.is_empty or analysis_roi.area <= 0.0:
        raise DataContractError("ROI does not overlap the raster footprint with positive area")
    if analysis_roi.geom_type not in {"Polygon", "MultiPolygon"}:
        raise DataContractError("ROI/raster intersection is not polygonal")

    metric_crs = choose_local_metric_crs(raster.crs, analysis_roi)
    location = analysis_roi.representative_point()
    gsd = pixel_size_in_metres(
        raster.transform,
        raster.crs,
        metric_crs,
        (float(location.x), float(location.y)),
    )
    raster = replace(raster, metric_crs=metric_crs, gsd=gsd)
    roi_metric = transform_geometry(analysis_roi, raster.crs, metric_crs)
    clipped = not analysis_roi.equals(roi_in_raster)
    mask_path = _validate_mask(invalid_mask_path, raster) if invalid_mask_path else None
    return ValidatedScene(
        raster=raster,
        roi_in_raster_crs=roi_in_raster,
        analysis_roi_in_raster_crs=analysis_roi,
        roi_in_metric_crs=roi_metric,
        roi_was_clipped=clipped,
        invalid_mask_path=mask_path,
    )
