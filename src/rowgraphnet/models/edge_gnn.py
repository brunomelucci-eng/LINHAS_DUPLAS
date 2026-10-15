"""Endpoint-graph edge scorer with an optional PyTorch Geometric backend."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

try:  # Keep PyG optional for CPU-only baseline and geometry installations.
    from torch_geometric.nn import GATv2Conv

    PYG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on environment
    GATv2Conv = None  # type: ignore[assignment]
    PYG_AVAILABLE = False


def _mlp(
    in_channels: int,
    out_channels: int,
    *,
    hidden_channels: int | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    hidden_channels = hidden_channels or out_channels
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_channels, out_channels),
    )


class _NativeMessageLayer(nn.Module):
    """Small index_add-based message layer requiring only core PyTorch."""

    def __init__(self, channels: int, *, edge_channels: int, dropout: float) -> None:
        super().__init__()
        self.message = _mlp(
            channels + edge_channels,
            channels,
            hidden_channels=channels,
            dropout=dropout,
        )
        self.update = _mlp(
            channels * 2,
            channels,
            hidden_channels=channels,
            dropout=dropout,
        )
        self.normalization = nn.LayerNorm(channels)

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None,
    ) -> Tensor:
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(nodes)
        else:
            source, destination = edge_index
            parts = [nodes[source]]
            if edge_features is not None:
                parts.append(edge_features)
            messages = self.message(torch.cat(parts, dim=-1))
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(0, destination, messages)
            degree = torch.zeros(
                nodes.shape[0],
                dtype=nodes.dtype,
                device=nodes.device,
            )
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=nodes.dtype))
            aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
        return self.normalization(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class _PyGMessageLayer(nn.Module):
    def __init__(self, channels: int, *, edge_channels: int, dropout: float) -> None:
        super().__init__()
        if GATv2Conv is None:  # pragma: no cover - guarded by EdgeGNN
            raise ImportError("torch-geometric is not installed")
        self.convolution = GATv2Conv(
            channels,
            channels,
            heads=1,
            concat=False,
            edge_dim=edge_channels or None,
            add_self_loops=False,
            dropout=dropout,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(channels)

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None,
    ) -> Tensor:
        update = self.convolution(nodes, edge_index, edge_attr=edge_features)
        return self.normalization(nodes + self.dropout(self.activation(update)))


class EdgeGNN(nn.Module):
    """Estimate connection probabilities for candidate endpoint pairs.

    ``backend='auto'`` uses PyG when available and otherwise selects a fully
    differentiable native PyTorch fallback. The public inputs and outputs are
    identical across backends.
    """

    def __init__(
        self,
        node_in_channels: int,
        *,
        edge_in_channels: int = 0,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        backend: Literal["auto", "pyg", "native"] = "auto",
        make_undirected: bool = True,
    ) -> None:
        super().__init__()
        if min(node_in_channels, hidden_channels, num_layers) < 1:
            raise ValueError("node, hidden and layer dimensions must be positive")
        if edge_in_channels < 0:
            raise ValueError("edge_in_channels cannot be negative")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if backend not in {"auto", "pyg", "native"}:
            raise ValueError("backend must be 'auto', 'pyg', or 'native'")
        if backend == "pyg" and not PYG_AVAILABLE:
            raise ImportError(
                "backend='pyg' requires torch-geometric; install the RowGraphNet 'gnn' extra"
            )

        self.backend = (
            "pyg" if backend == "pyg" or (backend == "auto" and PYG_AVAILABLE) else "native"
        )
        self.node_in_channels = int(node_in_channels)
        self.edge_in_channels = int(edge_in_channels)
        self.hidden_channels = int(hidden_channels)
        self.make_undirected = bool(make_undirected)

        self.node_encoder = _mlp(
            node_in_channels,
            hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.edge_encoder = (
            _mlp(
                edge_in_channels,
                hidden_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
            )
            if edge_in_channels
            else None
        )
        edge_channels = hidden_channels if edge_in_channels else 0
        layer_type = _PyGMessageLayer if self.backend == "pyg" else _NativeMessageLayer
        self.layers = nn.ModuleList(
            layer_type(hidden_channels, edge_channels=edge_channels, dropout=dropout)
            for _ in range(num_layers)
        )

        # Symmetric pair descriptors keep the final edge score invariant to
        # swapping the two endpoint indices.
        score_channels = hidden_channels * 3 + edge_channels
        self.edge_scorer = _mlp(
            score_channels,
            1,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return node embeddings, edge logits and edge probabilities."""

        self._validate_graph(node_features, edge_index, edge_features)
        encoded_nodes = self.node_encoder(node_features)
        encoded_edges = self.edge_encoder(edge_features) if self.edge_encoder is not None else None

        message_index = edge_index
        message_edges = encoded_edges
        if self.make_undirected and edge_index.shape[1] > 0:
            message_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
            if encoded_edges is not None:
                message_edges = torch.cat((encoded_edges, encoded_edges), dim=0)

        for layer in self.layers:
            encoded_nodes = layer(encoded_nodes, message_index, message_edges)

        if edge_index.shape[1] == 0:
            edge_logits = encoded_nodes.new_empty((0,))
        else:
            source, destination = edge_index
            source_nodes = encoded_nodes[source]
            destination_nodes = encoded_nodes[destination]
            descriptors = [
                source_nodes + destination_nodes,
                torch.abs(source_nodes - destination_nodes),
                source_nodes * destination_nodes,
            ]
            if encoded_edges is not None:
                descriptors.append(encoded_edges)
            edge_logits = self.edge_scorer(torch.cat(descriptors, dim=-1)).squeeze(-1)

        return {
            "node_embeddings": encoded_nodes,
            "edge_logits": edge_logits,
            "edge_probability": torch.sigmoid(edge_logits),
        }

    def _validate_graph(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None,
    ) -> None:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_in_channels:
            raise ValueError(f"node_features must have shape [nodes, {self.node_in_channels}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        edge_count = edge_index.shape[1]
        if edge_count:
            if node_features.shape[0] == 0:
                raise ValueError("edges cannot reference an empty node tensor")
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_features.shape[0]:
                raise ValueError("edge_index contains an out-of-range node index")

        if self.edge_in_channels:
            if edge_features is None:
                raise ValueError(f"edge_features with {self.edge_in_channels} columns are required")
            if edge_features.ndim != 2 or edge_features.shape != (
                edge_count,
                self.edge_in_channels,
            ):
                raise ValueError(f"edge_features must have shape [edges, {self.edge_in_channels}]")
        elif edge_features is not None and (
            edge_features.ndim != 2
            or edge_features.shape[0] != edge_count
            or edge_features.shape[1] != 0
        ):
            raise ValueError("this model was configured without edge features")
