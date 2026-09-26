import numpy as np
import geopandas as gpd
import rasterio
import src.postprocessing.regularized_centerline_fairing as fairing_module
from affine import Affine
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from src.postprocessing.regularized_centerline_fairing import (
    adaptive_resample_line,
    apply_class_hysteresis,
    classify_refinement_source,
    compute_dynamic_station_profile,
    compute_multiscale_angular_metrics,
    fair_centerlines_gdf,
    fair_line_regularized,
    repair_bad_interval_from_reference,
    resample_line,
    smooth_lateral_offsets,
    straighten_terminals_c1,
)


def fairing_config():
    return {
        "final_centerline_fairing": {
            "enabled": True,
            "sampling": {"longitudinal_spacing_m": 0.20},
            "angular_metrics": {
                "short_window_m": 0.25,
                "medium_window_m": 0.60,
                "long_window_m": 1.35,
            },
            "probability": {
                "min_probability": 0.12,
                "max_median_probability_loss": 0.02,
                "lateral_blur_sigma_samples": 1.0,
            },
            "regularization": {
                "first_difference_weight": 4.0,
                "second_difference_weight": 12.0,
                "original_distance_weight": 2.0,
                "pair_spacing_weight": 3.0,
            },
            "geometry": {
                "baseline_window_m": 1.40,
                "max_lateral_shift_m": 0.10,
                "max_hausdorff_m": 0.12,
                "preserve_endpoints": True,
                "require_simple": True,
            },
            "output_sampling": {
                "straight_spacing_m": 0.60,
                "curve_spacing_m": 0.25,
                "high_curvature_spacing_m": 0.18,
            },
            "splice_blending": {
                "enabled": True,
                "transition_length_m": 1.00,
                "require_c1_continuity": True,
            },
            "terminal": {
                "enabled": True,
                "trigger_distance_to_boundary_m": 2.0,
                "replace_terminal_m": 2.0,
                "direction_fit_window_m": 3.0,
                "transition_length_m": 0.50,
                "max_extension_m": 2.50,
                "use_sister_direction_when_available": True,
            },
        }
    }


def dynamic_fairing_config():
    config = fairing_config()
    fairing = config["final_centerline_fairing"]
    fairing["offset_smoothing"] = {
        "enabled": True,
        "method": "gaussian",
        "fallback_method": "moving_average",
        "dynamic_sigma_enabled": True,
        "moving_average_window_m": 0.80,
        "gaussian_sigma_m": 0.45,
        "straight_sigma_m": 0.70,
        "gentle_curve_sigma_m": 0.45,
        "curve_sigma_m": 0.28,
        "high_curvature_sigma_m": 0.18,
        "truncate": 3.0,
    }
    fairing["adaptive_output_sampling"] = {
        "enabled": True,
        "straight_spacing_m": 1.00,
        "gentle_curve_spacing_m": 0.60,
        "curve_spacing_m": 0.35,
        "high_curvature_spacing_m": 0.20,
        "terminal_spacing_m": 0.25,
        "splice_spacing_m": 0.20,
        "straight_max_turn_deg_per_m": 1.0,
        "gentle_curve_max_turn_deg_per_m": 3.0,
        "curve_max_turn_deg_per_m": 7.0,
        "min_class_persistence_m": 0.80,
        "transition_blend_m": 0.50,
    }
    return config


def circular_arc(radius_m, angle_rad=np.pi / 2, spacing_m=0.025):
    length = radius_m * angle_rad
    theta = np.linspace(0.0, angle_rad, int(np.ceil(length / spacing_m)) + 1)
    return LineString(
        np.column_stack(
            (radius_m * np.sin(theta), radius_m * (1.0 - np.cos(theta)))
        )
    )


def projected_steps(source, sampled):
    stations = np.asarray(
        [source.project(Point(coordinate)) for coordinate in sampled.coords]
    )
    return stations, np.diff(stations)


def center_probability_raster():
    transform = Affine.translation(-1.0, 1.0) * Affine.scale(0.02, -0.02)
    rows = np.arange(100, dtype=float)
    ys = 1.0 - (rows + 0.5) * 0.02
    profile = np.exp(-0.5 * (ys / 0.055) ** 2)
    return np.repeat(profile[:, None], 750, axis=1).astype(np.float32), transform


def test_multiscale_metrics_detect_short_staircase():
    x = np.arange(0.0, 12.01, 0.20)
    staircase = LineString(np.column_stack((x, 0.06 * ((np.arange(len(x)) % 2) * 2 - 1))))
    smooth = LineString(np.column_stack((x, np.zeros_like(x))))

    noisy_metrics = compute_multiscale_angular_metrics(staircase, fairing_config())
    smooth_metrics = compute_multiscale_angular_metrics(smooth, fairing_config())

    assert noisy_metrics.short.p95_turn_deg > smooth_metrics.short.p95_turn_deg + 10.0
    assert noisy_metrics.short.curvature_energy > smooth_metrics.short.curvature_energy
    assert noisy_metrics.short.lateral_inversions > 10


def test_vectorized_resampling_preserves_endpoints_and_length_axis():
    line = LineString([(0, 0), (2, 0), (3, 1), (5, 1)])
    distances, coords = resample_line(line, 0.20)
    assert np.array_equal(coords[0], np.asarray(line.coords[0], dtype=float))
    assert np.array_equal(coords[-1], np.asarray(line.coords[-1], dtype=float))
    assert distances[0] == 0.0
    assert abs(distances[-1] - line.length) < 1e-12


def test_moving_average_smooths_offsets_and_preserves_endpoints():
    config = fairing_config()
    config["final_centerline_fairing"]["offset_smoothing"] = {
        "enabled": True,
        "method": "moving_average",
        "moving_average_window_m": 0.80,
    }
    offsets = np.asarray([0.0, 0.08, -0.08, 0.08, -0.08, 0.08, 0.0])
    baseline = np.column_stack((np.arange(len(offsets)) * 0.20, np.zeros(len(offsets))))

    smoothed, metadata = smooth_lateral_offsets(
        offsets, baseline, 0.20, config
    )

    assert metadata["offset_smoothing_method"] == "moving_average"
    assert np.std(smoothed[1:-1]) < np.std(offsets[1:-1])
    assert smoothed[0] == offsets[0]
    assert smoothed[-1] == offsets[-1]


def test_gaussian_removes_staircase():
    config = dynamic_fairing_config()
    config["final_centerline_fairing"]["offset_smoothing"]["dynamic_sigma_enabled"] = False
    x = np.arange(0.0, 12.01, 0.20)
    offsets = 0.06 * ((np.arange(len(x)) % 2) * 2 - 1)
    baseline = np.column_stack((x, np.zeros_like(x)))

    smoothed, metadata = smooth_lateral_offsets(
        offsets, baseline, 0.20, config
    )
    before = LineString(np.column_stack((x, offsets)))
    after = LineString(np.column_stack((x, smoothed)))
    before_metrics = compute_multiscale_angular_metrics(before, config)
    after_metrics = compute_multiscale_angular_metrics(after, config)

    assert metadata["offset_smoothing_method"] == "gaussian"
    assert np.sqrt(np.mean(smoothed[1:-1] ** 2)) <= 0.25 * np.sqrt(
        np.mean(offsets[1:-1] ** 2)
    )
    assert after_metrics.short.p95_turn_deg <= 0.50 * before_metrics.short.p95_turn_deg
    assert smoothed[0] == offsets[0]
    assert smoothed[-1] == offsets[-1]


def test_gaussian_preserves_real_curve():
    config = dynamic_fairing_config()
    curve = circular_arc(12.0)
    _, baseline = resample_line(curve, 0.20)
    _, normals = fairing_module._tangents_and_normals(baseline)
    offsets = 0.035 * ((np.arange(len(baseline)) % 2) * 2 - 1)

    smoothed, metadata = smooth_lateral_offsets(
        offsets, baseline, 0.20, config
    )
    noisy = LineString(baseline + normals * offsets[:, None])
    result = LineString(baseline + normals * smoothed[:, None])

    assert metadata["offset_smoothing_method"] == "gaussian_dynamic"
    assert np.median(np.abs(smoothed[1:-1])) < 0.50 * np.median(
        np.abs(offsets[1:-1])
    )
    assert result.hausdorff_distance(LineString(baseline)) <= 0.035 + 1e-9
    assert result.coords[0] == noisy.coords[0]
    assert result.coords[-1] == noisy.coords[-1]


def test_dynamic_sigma_changes_with_curvature():
    config = dynamic_fairing_config()
    straight_profile = compute_dynamic_station_profile(
        LineString([(0.0, 0.0), (12.0, 0.0)]), config
    )
    curve_profile = compute_dynamic_station_profile(circular_arc(5.0), config)

    assert np.median(straight_profile.sigma_m[5:-5]) >= 0.69
    assert np.median(curve_profile.sigma_m[5:-5]) <= 0.20
    assert np.median(curve_profile.sigma_m[5:-5]) < np.median(
        straight_profile.sigma_m[5:-5]
    )


def test_straight_line_outputs_one_meter_vertices():
    config = dynamic_fairing_config()
    line = LineString([(0.0, 0.0), (10.0, 0.0)])

    result = adaptive_resample_line(line, config)
    stations, steps = projected_steps(line, result)

    assert len(result.coords) == 11
    assert np.allclose(stations, np.arange(0.0, 10.01, 1.0), atol=1e-9)
    assert np.allclose(steps, 1.0, atol=1e-9)
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def test_curve_receives_denser_vertices():
    config = dynamic_fairing_config()
    line = circular_arc(12.0)

    result = adaptive_resample_line(line, config)
    _, steps = projected_steps(line, result)
    interior_steps = steps[:-1] if len(steps) > 1 else steps

    assert 0.32 <= np.median(interior_steps) <= 0.38
    assert np.percentile(interior_steps, 95) <= 0.40
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def test_high_curvature_receives_minimum_spacing():
    config = dynamic_fairing_config()
    line = circular_arc(5.0)

    result = adaptive_resample_line(line, config)
    _, steps = projected_steps(line, result)
    interior_steps = steps[:-1] if len(steps) > 1 else steps

    assert 0.18 <= np.median(interior_steps) <= 0.22
    assert np.percentile(interior_steps, 95) <= 0.23
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def test_sampling_class_has_hysteresis():
    distances = np.arange(30, dtype=np.float64) * 0.20
    raw_classes = np.asarray(
        [0] * 10 + [2] * 3 + [0] * 10 + [2] * 7,
        dtype=np.int8,
    )

    stable = apply_class_hysteresis(raw_classes, distances, 0.80)

    assert np.all(stable[10:13] == 0)
    assert np.all(stable[23:30] == 2)


def test_endpoints_are_preserved():
    config = dynamic_fairing_config()
    x = np.arange(0.0, 12.01, 0.20)
    y = 3.0 + 0.05 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((2.0 + x, y)))

    sampled = adaptive_resample_line(line, config)
    result, _ = fair_line_regularized(line, config)

    assert sampled.coords[0] == line.coords[0]
    assert sampled.coords[-1] == line.coords[-1]
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]
    assert result.is_valid and result.is_simple


def test_short_line_fairing_does_not_crash():
    # Two or three stations previously produced candidate/reference arrays
    # with incompatible shapes inside the B-spline displacement safety check.
    lines = (
        LineString([(0.0, 0.0), (0.18, 0.02)]),
        LineString([(0.0, 0.0), (0.18, 0.025), (0.36, 0.0)]),
    )

    for line in lines:
        result, metadata = fair_line_regularized(line, fairing_config())

        assert result.is_valid
        assert result.is_simple
        assert len(result.coords) >= 2
        assert "fairing_reason" in metadata


def test_staircase_line_becomes_smooth():
    x = np.arange(0.0, 12.01, 0.20)
    y = 0.055 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((x, y)))
    probability, transform = center_probability_raster()
    before = compute_multiscale_angular_metrics(line, fairing_config())

    result, metadata = fair_line_regularized(
        line, fairing_config(), probability, transform
    )
    after = compute_multiscale_angular_metrics(result, fairing_config())

    assert metadata["fairing_applied"] is True
    assert after.short.p95_turn_deg <= before.short.p95_turn_deg * 0.50
    assert metadata["fairing_max_lateral_shift_m"] <= 0.10 + 1e-9
    assert line.hausdorff_distance(result) <= 0.12 + 1e-9
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def test_fairing_stays_inside_center_mass():
    x = np.arange(0.0, 12.01, 0.20)
    y = 0.075 + 0.02 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((x, y)))
    probability, transform = center_probability_raster()

    result, metadata = fair_line_regularized(
        line, fairing_config(), probability, transform
    )
    sampled = np.asarray(result.coords, dtype=float)

    assert metadata["fairing_applied"] is True
    assert abs(np.median(sampled[1:-1, 1])) < 0.06
    assert metadata["fairing_probability_after"] >= metadata["fairing_probability_before"] - 0.02


def test_line_stays_in_probability_center():
    config = dynamic_fairing_config()
    x = np.arange(0.0, 12.01, 0.20)
    y = 0.075 + 0.02 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((x, y)))
    probability, transform = center_probability_raster()

    result, metadata = fair_line_regularized(
        line, config, probability, transform
    )

    assert metadata["fairing_applied"] is True
    assert abs(np.median(np.asarray(result.coords)[1:-1, 1])) < 0.06
    assert metadata["fairing_probability_after"] >= metadata["fairing_probability_before"] - 0.02
    assert metadata["fairing_max_lateral_shift_m"] <= 0.10 + 1e-9
    assert metadata["fairing_hausdorff_m"] <= 0.12 + 1e-9


def test_fairing_preserves_real_curve():
    x = np.linspace(0.0, 16.0, 161)
    line = LineString(np.column_stack((x, 0.35 * np.sin(x / 4.0))))

    result, _ = fair_line_regularized(line, fairing_config())

    assert line.hausdorff_distance(result) <= 0.10 + 1e-9
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def _join_angle(line, distance, probe=0.08):
    before = np.asarray(line.interpolate(max(0.0, distance - probe)).coords[0], dtype=float)
    center = np.asarray(line.interpolate(distance).coords[0], dtype=float)
    after = np.asarray(line.interpolate(min(line.length, distance + probe)).coords[0], dtype=float)
    a, b = center - before, after - center
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominator == 0 else float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1))))


def test_reference_repair_uses_only_bad_interval():
    target = LineString([(0, 0), (4, 0), (4.5, 0.20), (5, -0.20), (5.5, 0.20), (6, 0), (10, 0)])
    reference = LineString([(0, 0.9), (10, 0.9)])
    interval_start = target.project(Point(4.0, 0.0))
    interval_end = target.project(Point(6.0, 0.0))

    repaired, metadata = repair_bad_interval_from_reference(
        target, reference, interval_start, interval_end, fairing_config()
    )

    assert metadata["splice_repaired"] is True
    assert substring(repaired, 0.0, 3.5).hausdorff_distance(substring(target, 0.0, 3.5)) < 1e-9
    repaired_suffix = substring(repaired, repaired.project(Point(6.5, 0)), repaired.length)
    target_suffix = substring(target, target.project(Point(6.5, 0)), target.length)
    assert repaired_suffix.hausdorff_distance(target_suffix) < 1e-9


def test_reference_repair_has_c1_splices():
    target = LineString([(0, 0), (4, 0), (4.5, 0.20), (5, -0.20), (5.5, 0.20), (6, 0), (10, 0)])
    reference = LineString([(0, 0.9), (5, 1.0), (10, 0.9)])
    interval_start = target.project(Point(4.0, 0.0))
    interval_end = target.project(Point(6.0, 0.0))
    repaired, metadata = repair_bad_interval_from_reference(
        target, reference, interval_start, interval_end, fairing_config()
    )
    assert metadata["splice_repaired"] is True
    assert _join_angle(repaired, repaired.project(Point(4.0, 0.0))) < 8.0
    assert _join_angle(repaired, repaired.project(Point(6.0, 0.0))) < 8.0


def test_sister_line_repairs_only_bad_interval():
    target = LineString(
        [(0, 0), (4, 0), (4.5, 0.20), (5, -0.20), (5.5, 0.20), (6, 0), (10, 0)]
    )
    sister = LineString([(0, 0.9), (10, 0.9)])
    repaired, metadata = repair_bad_interval_from_reference(
        target,
        sister,
        target.project(Point(4.0, 0.0)),
        target.project(Point(6.0, 0.0)),
        dynamic_fairing_config(),
    )

    assert metadata["splice_repaired"] is True
    assert substring(repaired, 0.0, 3.5).hausdorff_distance(
        substring(target, 0.0, 3.5)
    ) < 1e-9
    repaired_suffix = substring(
        repaired, repaired.project(Point(6.5, 0)), repaired.length
    )
    target_suffix = substring(target, target.project(Point(6.5, 0)), target.length)
    assert repaired_suffix.hausdorff_distance(target_suffix) < 1e-9


def test_both_broken_use_neighbor_only_as_template():
    first = LineString([(0, 0), (4, 0), (5, 0.25), (6, 0), (10, 0)])
    second = LineString([(0, -0.9), (4, -0.9), (5, -0.65), (6, -0.9), (10, -0.9)])
    template = LineString([(0, 0.9), (5, 1.0), (10, 0.9)])
    repaired_first, _ = repair_bad_interval_from_reference(first, template, 4, 6, fairing_config())
    repaired_second, _ = repair_bad_interval_from_reference(second, template, 4, 6, fairing_config())
    assert repaired_first.distance(template) > 0.5
    assert repaired_second.distance(template) > 1.2
    assert abs(repaired_first.distance(repaired_second) - 0.9) < 0.08


def test_terminal_last_two_meters_are_straight():
    roi = Polygon([(0, -2), (12, -2), (12, 2), (0, 2)])
    line = LineString([(1, 0), (8.5, 0), (9.5, 0.15), (10.5, -0.10), (11, 0.05)])

    result, metadata = straighten_terminals_c1(line, roi, fairing_config())
    tail = substring(result, max(0.0, result.length - 2.0), result.length)
    tail_metrics = compute_multiscale_angular_metrics(tail, fairing_config())

    assert metadata["fairing_terminal_changed"] is True
    assert metadata["fairing_terminal_extended_m"] <= 2.50 + 1e-9
    assert Point(result.coords[-1]).distance(roi.boundary) < 1e-7
    assert tail_metrics.short.p95_turn_deg < 1.0
    tail_steps = np.linalg.norm(np.diff(np.asarray(tail.coords), axis=0), axis=1)
    assert np.max(tail_steps) <= 0.25 + 1e-6


def test_terminal_blend_has_no_corner():
    roi = Polygon([(0, -2), (12, -2), (12, 2), (0, 2)])
    line = LineString([(1, 0), (8.0, 0), (9.0, 0.08), (10.0, 0.18), (11.0, 0.22)])
    result, metadata = straighten_terminals_c1(line, roi, fairing_config())
    assert metadata["fairing_terminal_changed"] is True
    anchor = max(0.0, result.length - 3.0)
    assert _join_angle(result, anchor) < 8.0


def test_terminal_extension_limit_is_reported_per_endpoint():
    roi = Polygon([(0, -2), (10, -2), (10, 2), (0, 2)])
    line = LineString([(2, 0), (5, 0), (8, 0)])

    result, metadata = straighten_terminals_c1(line, roi, fairing_config())

    assert metadata["fairing_terminal_changed"] is True
    assert metadata["fairing_terminal_extended_m"] <= 2.50 + 1e-9
    assert Point(result.coords[0]).distance(roi.boundary) < 1e-7
    assert Point(result.coords[-1]).distance(roi.boundary) < 1e-7


def test_good_line_is_unchanged():
    line = LineString([(1, 0), (10, 0)])
    gdf = gpd.GeoDataFrame({"quality_after": ["GOOD"]}, geometry=[line], crs="EPSG:3857")
    roi = Polygon([(0, -2), (12, -2), (12, 2), (0, 2)])
    config = dynamic_fairing_config()
    config["final_centerline_fairing"]["good_line_serration_override"] = {
        "enabled": True,
        "min_line_length_m": 3.0,
        "min_short_p95_deg": 2.50,
        "min_lateral_inversions_per_m": 0.75,
    }
    result, _ = fair_centerlines_gdf(gdf, roi, config)
    assert result.geometry.iloc[0].equals_exact(line, tolerance=0.0)
    assert result.loc[0, "fairing_reason"] == "good_preserved"
    assert result.loc[0, "fairing_good_serration_override"] == False


def test_serrated_good_line_overrides_preservation():
    x = np.arange(1.0, 11.01, 0.10)
    zigzag = 0.025 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((x, zigzag)))
    gdf = gpd.GeoDataFrame(
        {"quality_after": ["GOOD"]}, geometry=[line], crs="EPSG:3857"
    )
    roi = Polygon([(-4, -3), (16, -3), (16, 3), (-4, 3)])
    config = dynamic_fairing_config()
    fairing = config["final_centerline_fairing"]
    fairing["terminal"]["enabled"] = False
    fairing["good_line_serration_override"] = {
        "enabled": True,
        "min_line_length_m": 3.0,
        "min_short_p95_deg": 2.50,
        "min_lateral_inversions_per_m": 0.75,
    }

    before = compute_multiscale_angular_metrics(line, config)
    result, _ = fair_centerlines_gdf(gdf, roi, config)
    after_line = result.geometry.iloc[0]
    after = compute_multiscale_angular_metrics(after_line, config)

    assert result.loc[0, "fairing_good_serration_override"] == True
    assert result.loc[0, "fairing_applied"] == True
    assert not after_line.equals_exact(line, tolerance=1e-9)
    assert after.short.p95_turn_deg < before.short.p95_turn_deg * 0.50
    assert line.hausdorff_distance(after_line) <= 0.12 + 1e-9
    assert after_line.is_simple


def test_conservative_gaussian_fallback_salvages_rejected_candidate(monkeypatch):
    x = np.arange(0.0, 12.01, 0.10)
    zigzag = 0.05 * ((np.arange(len(x)) % 2) * 2 - 1)
    line = LineString(np.column_stack((x, zigzag)))
    config = dynamic_fairing_config()

    def unsafe_primary_spline(_optimized, original, _distances, _config):
        coordinates = np.asarray(original, dtype=np.float64).copy()
        coordinates[1:-1, 1] += 0.20
        return LineString(coordinates)

    monkeypatch.setattr(
        fairing_module, "_restricted_bspline", unsafe_primary_spline
    )

    result, metadata = fair_line_regularized(line, config)

    assert metadata["fairing_applied"] is True
    assert metadata["fairing_initial_rejection_reason"] in {
        "hausdorff_limit",
        "lateral_shift_limit",
    }
    assert metadata["fairing_fallback_method"] == "coordinate_gaussian"
    assert metadata["fairing_fallback_sigma_m"] == 0.45
    assert metadata["fairing_short_p95_after_deg"] < (
        metadata["fairing_short_p95_before_deg"] * 0.50
    )
    assert metadata["fairing_hausdorff_m"] <= 0.12 + 1e-9
    assert metadata["fairing_max_lateral_shift_m"] <= 0.10 + 1e-9
    assert result.coords[0] == line.coords[0]
    assert result.coords[-1] == line.coords[-1]


def test_pair_spacing_is_preserved():
    x = np.arange(0.0, 12.01, 0.20)
    zigzag = 0.05 * ((np.arange(len(x)) % 2) * 2 - 1)
    lines = [
        LineString(np.column_stack((x, zigzag))),
        LineString(np.column_stack((x, 0.9 + zigzag))),
    ]
    gdf = gpd.GeoDataFrame(
        {"quality_after": ["SUSPECT", "SUSPECT"], "pair_id": [1, 1]},
        geometry=lines,
        crs="EPSG:3857",
    )
    roi = Polygon([(-2, -3), (15, -3), (15, 3), (-2, 3)])
    result, _ = fair_centerlines_gdf(gdf, roi, dynamic_fairing_config())
    assert abs(result.geometry.iloc[0].distance(result.geometry.iloc[1]) - 0.9) < 0.05


def test_no_new_crossing():
    x = np.arange(0.0, 12.01, 0.20)
    zigzag = 0.075 + 0.02 * ((np.arange(len(x)) % 2) * 2 - 1)
    target = LineString(np.column_stack((x, zigzag)))
    obstacle = LineString([(6.0, -0.04), (6.0, 0.04)])
    gdf = gpd.GeoDataFrame(
        {"quality_after": ["SUSPECT", "GOOD"]},
        geometry=[target, obstacle],
        crs="EPSG:3857",
    )
    probability, transform = center_probability_raster()
    roi = Polygon([(-2, -3), (15, -3), (15, 3), (-2, 3)])
    result, _ = fair_centerlines_gdf(gdf, roi, fairing_config(), probability, transform)
    assert result.geometry.iloc[0].equals_exact(target, tolerance=0.0)


def test_no_crossing_after_dynamic_resampling():
    x = np.arange(0.0, 12.01, 0.20)
    zigzag = 0.075 + 0.02 * ((np.arange(len(x)) % 2) * 2 - 1)
    target = LineString(np.column_stack((x, zigzag)))
    obstacle = LineString([(6.0, -0.04), (6.0, 0.04)])
    gdf = gpd.GeoDataFrame(
        {"quality_after": ["SUSPECT", "GOOD"]},
        geometry=[target, obstacle],
        crs="EPSG:3857",
    )
    probability, transform = center_probability_raster()
    roi = Polygon([(-2, -3), (15, -3), (15, 3), (-2, 3)])

    result, _ = fair_centerlines_gdf(
        gdf, roi, dynamic_fairing_config(), probability, transform
    )

    assert result.geometry.iloc[0].equals_exact(target, tolerance=0.0)
    assert result.geometry.iloc[1].equals_exact(obstacle, tolerance=0.0)
    assert not result.geometry.iloc[0].intersects(result.geometry.iloc[1])
    assert result.loc[0, "fairing_applied"] == False
    assert result.loc[0, "fairing_reason"] == "new_network_contact"


def test_roi_clip_cannot_shorten_an_existing_valid_line(monkeypatch):
    original = LineString([(-1.0, 0.0), (5.0, 0.0)])
    unsafe = LineString([(-1.0, 0.0), (1.0, 0.0)])
    gdf = gpd.GeoDataFrame(
        {"quality_after": ["SUSPECT"]}, geometry=[original], crs="EPSG:3857"
    )
    roi = Polygon([(0, -1), (5, -1), (5, 1), (0, 1)])

    monkeypatch.setattr(
        fairing_module,
        "fair_line_regularized",
        lambda line, *args, **kwargs: (
            line,
            {"fairing_applied": False, "fairing_reason": "no_short_scale_gain"},
        ),
    )
    monkeypatch.setattr(
        fairing_module,
        "straighten_terminals_c1",
        lambda *args, **kwargs: (
            unsafe,
            {"fairing_terminal_changed": True, "fairing_terminal_reason": "accepted"},
        ),
    )

    result, _ = fair_centerlines_gdf(gdf, roi, fairing_config())

    assert result.geometry.iloc[0].equals_exact(original, tolerance=0.0)
    assert result.loc[0, "fairing_reason"] == "roi_clipped_too_short"


def test_metadata_source_is_not_nan_truthy():
    assert classify_refinement_source(
        {
            "reference_repair": np.nan,
            "kink_repair": np.nan,
            "terminal_straightened": np.nan,
            "smoothed": np.nan,
        }
    ) == "original"
    assert classify_refinement_source({"kink_repair": 1}) == "kink_repaired"
    assert classify_refinement_source(
        {"reference_repair": True, "fairing_terminal_changed": True}
    ) == "mixed"


def test_postprocess_only_cli_roundtrip(tmp_path):
    from scripts.refine_existing_vectors import main

    vectors_path = tmp_path / "vectors.gpkg"
    probability_path = tmp_path / "probability.tif"
    roi_path = tmp_path / "roi.geojson"
    output_path = tmp_path / "refined.gpkg"
    x = np.arange(1.0, 9.01, 0.20)
    zigzag = 0.04 * ((np.arange(len(x)) % 2) * 2 - 1)
    vectors = gpd.GeoDataFrame(
        {
            "quality_after": ["SUSPECT", "SUSPECT"],
            "pair_id": [1, 1],
            "model_name": ["test", "test"],
            "checkpoint_name": ["test.pt", "test.pt"],
        },
        geometry=[
            LineString(np.column_stack((x, zigzag))),
            LineString(np.column_stack((x, 0.9 + zigzag))),
        ],
        crs="EPSG:3857",
    )
    vectors.to_file(vectors_path, layer="predicted_rows", driver="GPKG")
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -2), (10, -2), (10, 3), (0, 3)])],
        crs="EPSG:3857",
    )
    roi.to_file(roi_path, driver="GeoJSON")
    transform = rasterio.transform.from_origin(0.0, 3.0, 0.05, 0.05)
    rows = np.arange(100, dtype=float)
    ys = 3.0 - (rows + 0.5) * 0.05
    probability = (
        np.exp(-0.5 * (ys / 0.08) ** 2)
        + np.exp(-0.5 * ((ys - 0.9) / 0.08) ** 2)
    ).astype(np.float32)
    with rasterio.open(
        probability_path,
        "w",
        driver="GTiff",
        width=200,
        height=100,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        tiled=True,
        blockxsize=32,
        blockysize=32,
    ) as dst:
        dst.write(np.repeat(probability[:, None], 200, axis=1), 1)

    exit_code = main(
        [
            "--config", "configs/roi_refinement_debug.yaml",
            "--input-vectors", str(vectors_path),
            "--probability-raster", str(probability_path),
            "--roi", str(roi_path),
            "--output", str(output_path),
        ]
    )

    result = gpd.read_file(output_path, layer="final_lines")
    assert exit_code == 0
    assert len(result) == 2
    assert "fairing_applied" in result.columns
    source = open("scripts/refine_existing_vectors.py", encoding="utf-8").read()
    for forbidden in ("RowPredictor", "predict_full_raster", "clean_mask", "perform_thinning", "skeleton_to_graph"):
        assert forbidden not in source
