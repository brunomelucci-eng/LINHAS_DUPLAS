"""Versioned JSON/YAML scene manifests with strict provenance validation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import yaml

SCHEMA_VERSION = 1
ALLOWED_SPLITS = frozenset({"train", "validation", "calibration", "test"})
SplitName = Literal["train", "validation", "calibration", "test"]


class ManifestError(ValueError):
    """Raised when a scene manifest violates the versioned data contract."""


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _string_values(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"{name} must be a sequence of strings")
    values = tuple(_non_empty_string(item, name=name) for item in value)
    if not values:
        raise ManifestError(f"{name} must not be empty")
    folded = [item.casefold() for item in values]
    if len(set(folded)) != len(folded):
        raise ManifestError(f"{name} contains duplicate values")
    return values


@dataclass(frozen=True, slots=True)
class VisibilitySpec:
    """Column/value contract distinguishing observed and occluded row pieces."""

    field: str = "visibility"
    observed_values: tuple[str, ...] = ("observed",)
    occluded_values: tuple[str, ...] = ("occluded",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _non_empty_string(self.field, name="visibility.field"))
        observed = _string_values(self.observed_values, name="visibility.observed_values")
        occluded = _string_values(self.occluded_values, name="visibility.occluded_values")
        if {value.casefold() for value in observed} & {value.casefold() for value in occluded}:
            raise ManifestError("observed and occluded visibility values must be disjoint")
        object.__setattr__(self, "observed_values", observed)
        object.__setattr__(self, "occluded_values", occluded)

    @classmethod
    def from_mapping(cls, value: Any) -> VisibilitySpec:
        if not isinstance(value, Mapping):
            raise ManifestError("visibility must be a mapping")
        return cls(
            field=value.get("field", "visibility"),
            observed_values=tuple(
                value.get("observed_values", value.get("observed", ("observed",)))
            ),
            occluded_values=tuple(
                value.get("occluded_values", value.get("occluded", ("occluded",)))
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "observed_values": list(self.observed_values),
            "occluded_values": list(self.occluded_values),
        }


@dataclass(frozen=True, slots=True)
class IgnoreSpec:
    """Optional external ignore asset plus annotation values treated as ignore."""

    field: str = "visibility"
    values: tuple[str, ...] = ("ignore",)
    path: Path | None = None
    layer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _non_empty_string(self.field, name="ignore.field"))
        object.__setattr__(self, "values", _string_values(self.values, name="ignore.values"))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        if self.layer is not None:
            object.__setattr__(self, "layer", _non_empty_string(self.layer, name="ignore.layer"))

    def to_mapping(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"field": self.field, "values": list(self.values)}
        if self.path is not None:
            result["path"] = _display_path(self.path, base_dir)
        if self.layer is not None:
            result["layer"] = self.layer
        return result


@dataclass(frozen=True, slots=True)
class SceneManifest:
    """One geospatial scene; splitting is assigned at its ``group_id``."""

    schema_version: int
    scene_id: str
    group_id: str
    split: SplitName
    image: Path
    bands: tuple[int, ...]
    roi: Path
    annotations: Path
    visibility: VisibilitySpec
    ignore: IgnoreSpec
    sensor: Mapping[str, Any]
    roi_layer: str | None = None
    annotations_layer: str | None = None
    row_id_field: str = "row_id"
    source_manifest: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        object.__setattr__(self, "scene_id", _non_empty_string(self.scene_id, name="scene_id"))
        object.__setattr__(self, "group_id", _non_empty_string(self.group_id, name="group_id"))
        if self.split not in ALLOWED_SPLITS:
            raise ManifestError(
                f"split must be one of {sorted(ALLOWED_SPLITS)}, got {self.split!r}"
            )
        object.__setattr__(self, "image", Path(self.image))
        object.__setattr__(self, "roi", Path(self.roi))
        object.__setattr__(self, "annotations", Path(self.annotations))
        if self.source_manifest is not None:
            object.__setattr__(self, "source_manifest", Path(self.source_manifest))
        normalized_bands: list[int] = []
        for band in self.bands:
            if isinstance(band, bool) or not isinstance(band, int) or band < 1:
                raise ManifestError("bands must contain positive one-based integers")
            normalized_bands.append(int(band))
        if not normalized_bands or len(set(normalized_bands)) != len(normalized_bands):
            raise ManifestError("bands must be non-empty and contain no duplicates")
        object.__setattr__(self, "bands", tuple(normalized_bands))
        object.__setattr__(
            self, "row_id_field", _non_empty_string(self.row_id_field, name="row_id_field")
        )
        if self.roi_layer is not None:
            object.__setattr__(
                self, "roi_layer", _non_empty_string(self.roi_layer, name="roi.layer")
            )
        if self.annotations_layer is not None:
            object.__setattr__(
                self,
                "annotations_layer",
                _non_empty_string(self.annotations_layer, name="annotations.layer"),
            )
        if not isinstance(self.sensor, Mapping):
            raise ManifestError("sensor must be a mapping")
        try:
            json.dumps(dict(self.sensor))
        except (TypeError, ValueError) as exc:
            raise ManifestError("sensor metadata must be JSON-serializable") from exc
        object.__setattr__(self, "sensor", dict(self.sensor))

    def to_mapping(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        image: dict[str, Any] = {"path": _display_path(self.image, base_dir)}
        roi: dict[str, Any] = {"path": _display_path(self.roi, base_dir)}
        annotations: dict[str, Any] = {
            "path": _display_path(self.annotations, base_dir),
            "row_id_field": self.row_id_field,
        }
        if self.roi_layer is not None:
            roi["layer"] = self.roi_layer
        if self.annotations_layer is not None:
            annotations["layer"] = self.annotations_layer
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "group_id": self.group_id,
            "split": self.split,
            "image": image,
            "bands": list(self.bands),
            "roi": roi,
            "annotations": annotations,
            "visibility": self.visibility.to_mapping(),
            "ignore": self.ignore.to_mapping(base_dir=base_dir),
            "sensor": dict(self.sensor),
        }


def _display_path(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:  # Different Windows drives cannot be made relative.
        return str(path)


def _resolve_path(
    value: Any, *, name: str, base_dir: Path
) -> tuple[Path, str | None, Mapping[str, Any]]:
    if isinstance(value, (str, PathLike)):
        raw_path = value
        options: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        if "path" not in value:
            raise ManifestError(f"{name} mapping must define path")
        raw_path = value["path"]
        options = value
    else:
        raise ManifestError(f"{name} must be a path string or mapping")
    if not isinstance(raw_path, (str, PathLike)) or not str(raw_path).strip():
        raise ManifestError(f"{name}.path must be a non-empty local path")
    if "://" in str(raw_path):
        raise ManifestError(f"{name}.path must be local, not a URL")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    layer = options.get("layer")
    if layer is not None:
        layer = _non_empty_string(layer, name=f"{name}.layer")
    return path, layer, options


def _parse_ignore(value: Any, *, base_dir: Path, visibility_field: str) -> IgnoreSpec:
    if not isinstance(value, Mapping):
        raise ManifestError("ignore must be a mapping")
    path: Path | None = None
    layer: str | None = None
    if value.get("path") is not None:
        path, layer, _ = _resolve_path(value, name="ignore", base_dir=base_dir)
    return IgnoreSpec(
        field=value.get("field", visibility_field),
        values=tuple(value.get("values", ("ignore",))),
        path=path,
        layer=layer,
    )


def scene_manifest_from_mapping(
    value: Mapping[str, Any],
    *,
    base_dir: str | PathLike[str] | Path = ".",
    check_paths: bool = True,
    source_manifest: Path | None = None,
) -> SceneManifest:
    """Parse and validate one already-decoded manifest mapping."""

    if not isinstance(value, Mapping):
        raise ManifestError("scene manifest root must be a mapping")
    required = {
        "schema_version",
        "scene_id",
        "group_id",
        "split",
        "image",
        "bands",
        "roi",
        "annotations",
        "visibility",
        "ignore",
        "sensor",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ManifestError(f"scene manifest is missing required fields: {missing}")
    root = Path(base_dir).expanduser().resolve()
    image, _, _ = _resolve_path(value["image"], name="image", base_dir=root)
    roi, roi_layer, _ = _resolve_path(value["roi"], name="roi", base_dir=root)
    annotations, annotations_layer, annotation_options = _resolve_path(
        value["annotations"],
        name="annotations",
        base_dir=root,
    )
    visibility = VisibilitySpec.from_mapping(value["visibility"])
    ignore = _parse_ignore(value["ignore"], base_dir=root, visibility_field=visibility.field)
    bands = value["bands"]
    if not isinstance(bands, Sequence) or isinstance(bands, (str, bytes)):
        raise ManifestError("bands must be a sequence")
    manifest = SceneManifest(
        schema_version=value["schema_version"],
        scene_id=value["scene_id"],
        group_id=value["group_id"],
        split=value["split"],
        image=image,
        bands=tuple(bands),
        roi=roi,
        annotations=annotations,
        visibility=visibility,
        ignore=ignore,
        sensor=value["sensor"],
        roi_layer=roi_layer,
        annotations_layer=annotations_layer,
        row_id_field=annotation_options.get("row_id_field", "row_id"),
        source_manifest=source_manifest,
    )
    _validate_paths(manifest, check_paths=check_paths)
    return manifest


def _validate_paths(manifest: SceneManifest, *, check_paths: bool) -> None:
    if manifest.image == manifest.roi or manifest.image == manifest.annotations:
        raise ManifestError("image path cannot also be used as ROI or annotations")
    if not check_paths:
        return
    assets = {
        "image": manifest.image,
        "roi": manifest.roi,
        "annotations": manifest.annotations,
    }
    if manifest.ignore.path is not None:
        assets["ignore"] = manifest.ignore.path
    for name, path in assets.items():
        if not path.is_file():
            raise ManifestError(f"{name} path does not exist or is not a file: {path}")


def _read_document(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read manifest: {path}") from exc
    try:
        if path.suffix.casefold() == ".json":
            value = json.loads(text)
        elif path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            raise ManifestError("manifest extension must be .json, .yaml, or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"manifest is not valid {path.suffix}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("scene manifest root must be a mapping")
    return value


def load_scene_manifest(
    manifest_path: str | PathLike[str],
    *,
    check_paths: bool = True,
) -> SceneManifest:
    """Load one versioned JSON/YAML scene manifest."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"manifest path does not exist or is not a file: {path}")
    return scene_manifest_from_mapping(
        _read_document(path),
        base_dir=path.parent,
        check_paths=check_paths,
        source_manifest=path,
    )


def validate_scene_manifests(scenes: Iterable[SceneManifest]) -> tuple[SceneManifest, ...]:
    """Reject duplicate scenes/assets and cross-split group leakage."""

    result = tuple(scenes)
    scene_ids: set[str] = set()
    signatures: set[tuple[Any, ...]] = set()
    for scene in result:
        if scene.scene_id in scene_ids:
            raise ManifestError(f"duplicate scene_id: {scene.scene_id!r}")
        scene_ids.add(scene.scene_id)
        signature = (
            scene.image,
            scene.roi,
            scene.roi_layer,
            scene.annotations,
            scene.annotations_layer,
            scene.bands,
        )
        if signature in signatures:
            raise ManifestError(f"duplicate scene assets detected for scene_id {scene.scene_id!r}")
        signatures.add(signature)
    from rowgraphnet.data.splits import validate_group_isolation

    validate_group_isolation(result)
    return result


def load_scene_manifests(
    manifest_paths: Iterable[str | PathLike[str]],
    *,
    check_paths: bool = True,
) -> tuple[SceneManifest, ...]:
    """Load several per-scene files and validate their dataset-wide invariants."""

    paths = [Path(path).expanduser().resolve() for path in manifest_paths]
    if len(set(paths)) != len(paths):
        raise ManifestError("manifest path list contains duplicates")
    return validate_scene_manifests(
        load_scene_manifest(path, check_paths=check_paths) for path in paths
    )


def write_scene_manifest(
    scene: SceneManifest,
    manifest_path: str | PathLike[str],
    *,
    relative_paths: bool = True,
) -> Path:
    """Write one canonical JSON/YAML manifest, preserving schema version."""

    path = Path(manifest_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = scene.to_mapping(base_dir=path.parent if relative_paths else None)
    suffix = path.suffix.casefold()
    if suffix == ".json":
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    elif suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    else:
        raise ManifestError("manifest extension must be .json, .yaml, or .yml")
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not write manifest: {path}") from exc
    return path
