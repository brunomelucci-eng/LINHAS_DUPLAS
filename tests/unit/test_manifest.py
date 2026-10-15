from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rowgraphnet.data import (
    IgnoreSpec,
    ManifestError,
    SceneManifest,
    SplitFractions,
    SplitLeakageError,
    VisibilitySpec,
    assign_group_splits,
    generate_group_split_map,
    load_scene_manifest,
    load_scene_manifests,
    validate_group_isolation,
    write_scene_manifest,
)


def _assets(tmp_path):
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "image": "ortho.tif",
            "roi": "vectors.gpkg",
            "annotations": "rows.gpkg",
            "ignore": "invalid.tif",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"test fixture")
    return paths


def _mapping(tmp_path, *, scene_id="scene-1", group_id="field-1", split="train"):
    _assets(tmp_path)
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "group_id": group_id,
        "split": split,
        "image": {"path": "ortho.tif"},
        "bands": [1, 2, 3],
        "roi": {"path": "vectors.gpkg", "layer": "roi"},
        "annotations": {
            "path": "rows.gpkg",
            "layer": "centerlines",
            "row_id_field": "row_id",
        },
        "visibility": {
            "field": "visibility",
            "observed_values": ["observed"],
            "occluded_values": ["occluded"],
        },
        "ignore": {
            "field": "visibility",
            "values": ["ignore"],
            "path": "invalid.tif",
        },
        "sensor": {"name": "synthetic", "platform": "unit-test"},
    }


def _scene(tmp_path, index, group_id, split="train"):
    image = tmp_path / f"image-{index}.tif"
    roi = tmp_path / f"roi-{index}.gpkg"
    annotations = tmp_path / f"rows-{index}.gpkg"
    return SceneManifest(
        schema_version=1,
        scene_id=f"scene-{index}",
        group_id=group_id,
        split=split,
        image=image,
        bands=(1, 2, 3),
        roi=roi,
        annotations=annotations,
        visibility=VisibilitySpec(),
        ignore=IgnoreSpec(),
        sensor={"name": "synthetic"},
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_versioned_json_yaml_manifest_resolves_paths_and_roundtrips(tmp_path, suffix):
    mapping = _mapping(tmp_path)
    path = tmp_path / f"scene{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(mapping), encoding="utf-8")
    else:
        import yaml

        path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    scene = load_scene_manifest(path)

    assert scene.schema_version == 1
    assert scene.image == (tmp_path / "ortho.tif").resolve()
    assert scene.roi_layer == "roi"
    assert scene.annotations_layer == "centerlines"
    assert scene.bands == (1, 2, 3)
    assert scene.ignore.path == (tmp_path / "invalid.tif").resolve()
    assert scene.sensor["name"] == "synthetic"

    roundtrip_path = tmp_path / f"roundtrip{suffix}"
    write_scene_manifest(scene, roundtrip_path)
    assert load_scene_manifest(roundtrip_path) == replace(
        scene, source_manifest=roundtrip_path.resolve()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("split", "dev", "split"),
        ("bands", [1, 1, 3], "duplicates"),
        ("bands", [0, 2, 3], "positive"),
    ],
)
def test_manifest_rejects_unsupported_version_split_and_bands(tmp_path, field, value, message):
    mapping = _mapping(tmp_path)
    mapping[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ManifestError, match=message):
        load_scene_manifest(path)


def test_manifest_requires_existing_paths_and_disjoint_visibility_values(tmp_path):
    mapping = _mapping(tmp_path)
    (tmp_path / "ortho.tif").unlink()
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ManifestError, match="image path"):
        load_scene_manifest(path)

    mapping = _mapping(tmp_path)
    mapping["visibility"]["occluded_values"] = ["OBSERVED"]
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ManifestError, match="disjoint"):
        load_scene_manifest(path)


def test_collection_rejects_duplicate_scene_ids_and_manifest_paths(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps(_mapping(tmp_path)), encoding="utf-8")
    second.write_text(json.dumps(_mapping(tmp_path)), encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate scene_id"):
        load_scene_manifests([first, second])
    with pytest.raises(ManifestError, match="path list contains duplicates"):
        load_scene_manifests([first, first])


def test_group_leakage_is_rejected_but_multiple_scenes_in_one_split_are_valid(tmp_path):
    scenes = (
        _scene(tmp_path, 1, "property-a", "train"),
        _scene(tmp_path, 2, "property-a", "train"),
    )
    assert validate_group_isolation(scenes) == {"property-a": "train"}

    leaked = (*scenes, _scene(tmp_path, 3, "property-a", "test"))
    with pytest.raises(SplitLeakageError, match="leaks across splits"):
        validate_group_isolation(leaked)


def test_group_split_is_deterministic_balanced_and_independent_of_scene_order(tmp_path):
    scenes = tuple(
        _scene(tmp_path, group * 2 + member, f"field-{group}")
        for group in range(10)
        for member in range(2)
    )
    fractions = SplitFractions(train=0.5, validation=0.2, calibration=0.1, test=0.2)

    assigned = assign_group_splits(scenes, fractions=fractions, seed=2026)
    reversed_assignment = assign_group_splits(reversed(scenes), fractions=fractions, seed=2026)

    by_scene = {scene.scene_id: scene.split for scene in assigned}
    assert by_scene == {scene.scene_id: scene.split for scene in reversed_assignment}
    by_group = validate_group_isolation(assigned)
    assert {split: list(by_group.values()).count(split) for split in fractions.as_mapping()} == {
        "train": 5,
        "validation": 2,
        "calibration": 1,
        "test": 2,
    }
    assert all(
        len({scene.split for scene in assigned if scene.group_id == group_id}) == 1
        for group_id in by_group
    )


def test_group_split_map_never_depends_on_duplicate_scene_or_tile_count():
    unique = generate_group_split_map(["a", "b", "c", "d"], seed="stable")
    repeated = generate_group_split_map(
        ["a", "a", "a", "b", "c", "c", "d"],
        seed="stable",
    )

    assert repeated == unique
