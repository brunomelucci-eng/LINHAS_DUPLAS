from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from rowgraphnet.models import (
    PYG_AVAILABLE,
    ConnectivityRefiner,
    EdgeGNN,
    MultiTaskDecoder,
    RowGraphNet,
    SegFormerEncoder,
)


def _tiny_segformer_config():
    transformers = pytest.importorskip("transformers")
    return transformers.SegformerConfig(
        num_channels=3,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 1, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[2, 2, 2, 2],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        output_hidden_states=True,
        return_dict=True,
    )


def test_real_transformers_encoder_returns_multiscale_maps() -> None:
    encoder = SegFormerEncoder(pretrained=False, config=_tiny_segformer_config())
    image = torch.randn(2, 3, 64, 80)

    features = encoder(image)

    assert encoder.feature_channels == (8, 16, 32, 64)
    assert len(features) == 4
    assert [feature.shape[:2] for feature in features] == [
        (2, 8),
        (2, 16),
        (2, 32),
        (2, 64),
    ]
    for fine, coarse in pairwise(features):
        assert coarse.shape[-2] <= fine.shape[-2]
        assert coarse.shape[-1] <= fine.shape[-1]


def test_rowgraphnet_outputs_input_sized_normalized_maps_on_cpu() -> None:
    model = RowGraphNet(
        pretrained=False,
        encoder_config=_tiny_segformer_config(),
        decoder_channels=16,
        refiner_hidden_channels=8,
    ).cpu()
    model.eval()
    image = torch.randn(1, 3, 65, 73)

    with torch.no_grad():
        output = model(image)

    one_channel_maps = (
        "band",
        "center",
        "center_observed",
        "center_complete",
        "endpoints",
        "distance",
        "uncertainty",
        "orientation_sin2",
        "orientation_cos2",
        "repair_probability",
    )
    for name in one_channel_maps:
        assert output[name].shape == (1, 1, 65, 73)
        assert torch.isfinite(output[name]).all()
    assert output["orientation"].shape == (1, 2, 65, 73)
    orientation_norm = torch.linalg.vector_norm(output["orientation"], dim=1)
    torch.testing.assert_close(
        orientation_norm, torch.ones_like(orientation_norm), atol=1e-5, rtol=1e-5
    )
    for name in ("band", "center", "endpoints", "repair_probability"):
        assert torch.all((output[name] >= 0) & (output[name] <= 1))
    assert torch.all(output["distance"] >= 0)
    assert torch.all(output["uncertainty"] > 0)
    # The zero-initialized residual refiner must start as an identity mapping.
    torch.testing.assert_close(output["center"], output["coarse_center"])
    torch.testing.assert_close(output["center_observed"], output["coarse_center"])
    torch.testing.assert_close(output["center_complete"], output["center"])


def test_decoder_has_deterministic_unit_orientation_for_zero_raw_vector() -> None:
    decoder = MultiTaskDecoder((4, 8), decoder_channels=8)
    torch.nn.init.zeros_(decoder.orientation_head.weight)
    torch.nn.init.zeros_(decoder.orientation_head.bias)
    features = (torch.randn(1, 4, 8, 9), torch.randn(1, 8, 4, 5))

    output = decoder(features, output_size=(31, 35))

    assert torch.count_nonzero(output["orientation_sin2"]) == 0
    torch.testing.assert_close(output["orientation_cos2"], torch.ones(1, 1, 31, 35))


def test_connectivity_refiner_is_initially_identity_and_backpropagates() -> None:
    refiner = ConnectivityRefiner(context_channels=4, hidden_channels=8)
    center_logits = torch.randn(2, 1, 12, 10, requires_grad=True)
    orientation = torch.nn.functional.normalize(torch.randn(2, 2, 12, 10), dim=1)
    endpoint_logits = torch.randn(2, 1, 12, 10)
    uncertainty = torch.rand(2, 1, 12, 10)
    context = torch.randn(2, 4, 6, 5)

    output = refiner(
        center_logits,
        orientation,
        endpoint_logits,
        uncertainty,
        context=context,
    )

    torch.testing.assert_close(output["refined_center_logits"], center_logits)
    output["refined_center"].sum().backward()
    assert center_logits.grad is not None
    assert torch.isfinite(center_logits.grad).all()


def test_native_edge_gnn_scores_candidates_and_empty_graphs() -> None:
    model = EdgeGNN(
        5,
        edge_in_channels=3,
        hidden_channels=16,
        num_layers=2,
        dropout=0.0,
        backend="native",
    )
    nodes = torch.randn(4, 5, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_features = torch.randn(3, 3)

    output = model(nodes, edge_index, edge_features)

    assert output["node_embeddings"].shape == (4, 16)
    assert output["edge_logits"].shape == (3,)
    assert output["edge_probability"].shape == (3,)
    assert torch.all((output["edge_probability"] >= 0) & (output["edge_probability"] <= 1))
    output["edge_logits"].sum().backward()
    assert nodes.grad is not None

    empty = model(
        torch.empty(0, 5),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 3),
    )
    assert empty["node_embeddings"].shape == (0, 16)
    assert empty["edge_logits"].shape == (0,)


def test_requesting_missing_pyg_has_actionable_error() -> None:
    if PYG_AVAILABLE:
        pytest.skip("PyG is installed in this environment")
    with pytest.raises(ImportError, match="torch-geometric"):
        EdgeGNN(4, backend="pyg")
