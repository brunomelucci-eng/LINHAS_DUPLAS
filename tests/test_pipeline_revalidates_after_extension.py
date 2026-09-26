from pathlib import Path


def test_prediction_pipeline_orders_final_topology_after_extension_and_clip():
    source = Path("src/postprocessing/postprocess_pipeline.py").read_text(
        encoding="utf-8"
    )
    preliminary_pairing = source.index(
        "lines = validate_double_rows(lines, working_config, preliminary=True)"
    )
    refinement = source.index("refined, refinement_debug = refine_double_rows")
    fairing = source.index("faired, fairing_debug = fair_centerlines_gdf")
    extension = source.index("extended = extend_lines_to_roi")
    clipping = source.index("clipped = clip_predictions_to_roi", extension)
    topology = source.index("topologically_valid, topology_rejected = _split_topology", clipping)
    pairing = source.index("final = validate_double_rows(topologically_valid", topology)
    probability_sampling = source.index("final = sample_line_probabilities", pairing)
    assert (
        preliminary_pairing
        < refinement
        < fairing
        < extension
        < clipping
        < topology
        < pairing
        < probability_sampling
    )
