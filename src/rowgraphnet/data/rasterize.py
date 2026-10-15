"""Dynamic rasterization of row identities, visibility, and ignore regions."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from rowgraphnet.data.contracts import (
    DataContractError,
    PixelSize,
    choose_local_metric_crs,
    is_metric_crs,
    pixel_size_in_metres,
    transform_geometry,
)


@dataclass(frozen=True, slots=True)
class RasterizedAnnotations:
    """Aligned targets with explicit observed/complete center semantics.

    ``center_observed`` contains only directly visible centerline pixels,
    whereas ``center_complete`` also contains annotated occlusions. The
    observed loss must use ``center_observed_valid_mask`` so an occluded pixel
    is ignored instead of becoming a false negative. ``center_complete`` and
    ``row_id_mask`` use ``center_complete_valid_mask``. Label zero is always
    background; positive labels are stable per-row instances.

    ``center_mask`` and ``observed_mask`` remain stored under their original
    names for constructor and serialization compatibility. The canonical
    training names below are zero-copy properties.
    """

    center_mask: np.ndarray
    row_id_mask: np.ndarray
    observed_mask: np.ndarray
    occluded_mask: np.ndarray
    ignore_mask: np.ndarray
    valid_mask: np.ndarray
    row_id_to_label: dict[Any, int]
    label_to_row_id: dict[int, Any]
    target_width_m: float
    gsd: PixelSize

    @property
    def center_observed(self) -> np.ndarray:
        """Visible center target for Focal + Dice supervision."""

        return self.observed_mask

    @property
    def center_complete(self) -> np.ndarray:
        """Full center target, including occlusion, for soft-clDice."""

        return self.center_mask

    @property
    def center_observed_valid_mask(self) -> np.ndarray:
        """Valid visible supervision, excluding occlusion/ignore/nodata."""

        return self.valid_mask & ~self.occluded_mask

    @property
    def center_complete_valid_mask(self) -> np.ndarray:
        """Valid complete-center supervision, excluding ignore/nodata."""

        return self.valid_mask

    def center_loss_targets(self) -> dict[str, np.ndarray]:
        """Return canonical tensors expected by :class:`RowGraphNetLoss`.

        Arrays are returned by reference. Dataset code may convert them to
        tensors, but must preserve the boolean masks and integer row labels.
        """

        return {
            "center_observed": self.center_observed,
            "center_complete": self.center_complete,
            "center_observed_valid_mask": self.center_observed_valid_mask,
            "center_complete_valid_mask": self.center_complete_valid_mask,
            "occluded_mask": self.occluded_mask,
            "row_id_mask": self.row_id_mask,
            "valid_mask": self.valid_mask,
        }


def _load_annotations(
    annotations: str | PathLike[str] | gpd.GeoDataFrame,
    *,
    allow_empty: bool = False,
) -> gpd.GeoDataFrame:
    if isinstance(annotations, (str, PathLike)):
        try:
            frame = gpd.read_file(annotations)
        except Exception as exc:
            raise DataContractError(f"could not read annotations: {annotations}") from exc
    elif isinstance(annotations, gpd.GeoDataFrame):
        frame = annotations.copy()
    else:
        raise DataContractError(f"unsupported annotations input type: {type(annotations).__name__}")
    if frame.empty and not allow_empty:
        raise DataContractError("annotations contain no features")
    if frame.crs is None:
        raise DataContractError("annotations must define a CRS")
    if not frame.empty:
        frame = frame.loc[frame.geometry.notna() & (~frame.geometry.is_empty)].copy()
        if not frame.geometry.is_valid.all():
            from shapely import make_valid

            frame["geometry"] = frame.geometry.apply(
                lambda g: make_valid(g) if g is not None and not g.is_valid else g
            )
            valid_geom = (
                frame.geometry.is_valid
                & (~frame.geometry.is_empty)
                & frame.geometry.geom_type.isin({"LineString", "MultiLineString"})
                & (frame.geometry.length > 0)
            )
            frame = frame.loc[valid_geom].copy()
            if frame.empty and not allow_empty:
                raise DataContractError("annotations contain invalid geometries")
    return frame


def _raster_footprint(transform: Affine, out_shape: tuple[int, int]) -> Polygon:
    height, width = out_shape
    return Polygon(
        [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
        ]
    )


def _burn(
    geometries: list[BaseGeometry],
    *,
    out_shape: tuple[int, int],
    transform: Affine,
    all_touched: bool,
) -> np.ndarray:
    if not geometries:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def _buffer_for_target(
    geometry: BaseGeometry,
    source_crs: CRS,
    metric_crs: CRS,
    width_m: float,
) -> BaseGeometry:
    metric_geometry = transform_geometry(geometry, source_crs, metric_crs)
    if metric_geometry.geom_type in {"LineString", "MultiLineString"}:
        metric_geometry = metric_geometry.buffer(
            width_m / 2.0, cap_style="round", join_style="round"
        )
    return transform_geometry(metric_geometry, metric_crs, source_crs)


def _stable_row_ids(values: list[Any]) -> list[Any]:
    unique: dict[tuple[str, str], Any] = {}
    for value in values:
        unique[(type(value).__name__, repr(value))] = value
    return [unique[key] for key in sorted(unique)]


def rasterize_row_annotations(
    annotations: str | PathLike[str] | gpd.GeoDataFrame,
    *,
    out_shape: tuple[int, int],
    transform: Affine,
    raster_crs: Any,
    row_id_field: str = "row_id",
    visibility_field: str = "visibility",
    occluded_values: tuple[str, ...] | list[str] = ("occluded",),
    ignore_values: tuple[str, ...] | list[str] = ("ignore",),
    centerline_width_m: float | None = None,
    metric_crs: Any | None = None,
    all_touched: bool = False,
    valid_mask: np.ndarray | None = None,
    allow_empty: bool = True,
) -> RasterizedAnnotations:
    """Rasterize complete per-row targets while retaining visibility semantics.

    Occluded centerlines remain positive complete-center/row targets and are
    identified in ``occluded_mask``. They are absent from ``center_observed``
    but excluded by ``center_observed_valid_mask``, so they are never learned
    as background. Explicit ignore regions and collisions between different
    ``row_id`` values override all supervision. ``valid_mask`` may additionally
    provide the raster/ROI eligibility grid: pass False for nodata and pixels
    outside the ROI. A metric target width is converted using the local GSD;
    when omitted, one local mean pixel is used rather than a fixed field unit.
    """

    if (
        len(out_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in out_shape
        )
        or any(value < 1 for value in out_shape)
    ):
        raise DataContractError("out_shape must contain positive integer height and width")
    shape = int(out_shape[0]), int(out_shape[1])
    if valid_mask is None:
        raster_valid = np.ones(shape, dtype=bool)
    else:
        raw_valid = np.asarray(valid_mask)
        if raw_valid.shape != shape:
            raise DataContractError(
                f"valid_mask shape {raw_valid.shape} does not match out_shape {shape}"
            )
        if np.issubdtype(raw_valid.dtype, np.number) and not np.isfinite(raw_valid).all():
            raise DataContractError("valid_mask must contain finite values")
        raster_valid = raw_valid.astype(bool, copy=False)
    frame = _load_annotations(annotations, allow_empty=allow_empty)
    try:
        source_crs = CRS.from_user_input(raster_crs)
    except Exception as exc:
        raise DataContractError(f"raster_crs is invalid: {raster_crs!r}") from exc
    if not frame.empty:
        frame = frame.to_crs(source_crs)
    if visibility_field not in frame.columns:
        frame[visibility_field] = "observed"

    normalized_visibility = frame[visibility_field].fillna("observed").astype(str).str.lower()
    occluded_set = {str(value).lower() for value in occluded_values}
    ignore_set = {str(value).lower() for value in ignore_values}
    is_ignore = normalized_visibility.isin(ignore_set)
    active = frame.loc[~is_ignore]
    if not active.empty and row_id_field not in frame.columns:
        raise DataContractError(f"annotations are missing required field {row_id_field!r}")
    if not active.empty and active[row_id_field].isna().any():
        raise DataContractError("non-ignore annotations must define row_id")
    invalid_row_geometry = ~active.geometry.geom_type.isin({"LineString", "MultiLineString"})
    if invalid_row_geometry.any():
        raise DataContractError("non-ignore row annotations must be LineString/MultiLineString")

    footprint = _raster_footprint(transform, shape)
    local_metric = (
        CRS.from_user_input(metric_crs)
        if metric_crs is not None
        else choose_local_metric_crs(source_crs, footprint)
    )
    if not is_metric_crs(local_metric):
        raise DataContractError("metric_crs must be projected with metre units")
    point = footprint.representative_point()
    gsd = pixel_size_in_metres(
        transform,
        source_crs,
        local_metric,
        (float(point.x), float(point.y)),
    )
    width_m = gsd.mean_m if centerline_width_m is None else float(centerline_width_m)
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise DataContractError("centerline_width_m must be finite and positive")

    buffered = [
        _buffer_for_target(geometry, source_crs, local_metric, width_m)
        for geometry in frame.geometry
    ]
    frame = frame.assign(_target_geometry=buffered, _visibility=normalized_visibility)

    ignore_geometries = frame.loc[is_ignore, "_target_geometry"].tolist()
    explicit_ignore = _burn(
        ignore_geometries,
        out_shape=shape,
        transform=transform,
        all_touched=all_touched,
    )

    row_ids = _stable_row_ids(active[row_id_field].tolist()) if not active.empty else []
    row_id_to_label = {row_id: label for label, row_id in enumerate(row_ids, start=1)}
    label_to_row_id = {label: row_id for row_id, label in row_id_to_label.items()}
    row_id_mask = np.zeros(shape, dtype=np.int32)
    observed_mask = np.zeros(shape, dtype=bool)
    occluded_mask = np.zeros(shape, dtype=bool)
    collisions = np.zeros(shape, dtype=bool)

    for row_id in row_ids:
        subset = frame.loc[(~is_ignore) & (frame[row_id_field] == row_id)]
        row_occluded = subset["_visibility"].isin(occluded_set)
        observed = _burn(
            subset.loc[~row_occluded, "_target_geometry"].tolist(),
            out_shape=shape,
            transform=transform,
            all_touched=all_touched,
        )
        occluded = _burn(
            subset.loc[row_occluded, "_target_geometry"].tolist(),
            out_shape=shape,
            transform=transform,
            all_touched=all_touched,
        )
        row_mask = observed | occluded
        collisions |= row_mask & (row_id_mask != 0)
        row_id_mask[row_mask & (row_id_mask == 0)] = row_id_to_label[row_id]
        observed_mask |= observed
        occluded_mask |= occluded

    # Occlusion is the more conservative interpretation wherever annotations overlap.
    observed_mask &= ~occluded_mask
    ignore_mask = explicit_ignore | collisions | ~raster_valid
    center_mask = observed_mask | occluded_mask
    center_mask[ignore_mask] = False
    observed_mask[ignore_mask] = False
    occluded_mask[ignore_mask] = False
    row_id_mask[ignore_mask] = 0
    supervision_valid_mask = ~ignore_mask
    return RasterizedAnnotations(
        center_mask=center_mask,
        row_id_mask=row_id_mask,
        observed_mask=observed_mask,
        occluded_mask=occluded_mask,
        ignore_mask=ignore_mask,
        valid_mask=supervision_valid_mask,
        row_id_to_label=row_id_to_label,
        label_to_row_id=label_to_row_id,
        target_width_m=width_m,
        gsd=gsd,
    )
