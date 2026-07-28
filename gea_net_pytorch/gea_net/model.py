from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def symmetric_normalize(adjacency: Tensor, eps: float = 1e-8) -> Tensor:
    """D^(-1/2) A D^(-1/2) for a batch of non-negative adjacency matrices."""
    degree = adjacency.sum(dim=-1).clamp_min(eps)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)


class TemporalStem(nn.Module):
    """Map raw scalar/vector samples to the d-dimensional sequence in the paper."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("temporal kernel_size must be odd")
        self.conv = nn.Conv1d(
            input_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B*N, T, input_dim]
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm(F.gelu(x)))


class ExternalAttention(nn.Module):
    """External attention with shared key/value memories and stable normalization."""

    def __init__(
        self,
        embed_dim: int,
        memory_size: int,
        dropout: float = 0.1,
        normalization_steps: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.normalization_steps = normalization_steps
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.memory_key = nn.Parameter(torch.empty(memory_size, embed_dim))
        self.memory_value = nn.Parameter(torch.empty(memory_size, embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.memory_key)
        nn.init.xavier_uniform_(self.memory_value)
        nn.init.xavier_uniform_(self.query.weight)

    def _normalize_attention(self, logits: Tensor) -> Tensor:
        # Raw L2-normalized logits can be negative and are not attention
        # probabilities. Softmax plus alternating column/row normalization is
        # non-negative, stable, and retains row sums of one.
        attention = logits.softmax(dim=-1)
        for _ in range(self.normalization_steps):
            attention = attention / attention.sum(dim=-2, keepdim=True).clamp_min(1e-8)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return attention

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        query = self.query(sequence)
        logits = torch.matmul(query, self.memory_key.t()) / math.sqrt(self.embed_dim)
        attention = self._normalize_attention(logits)
        recalled = torch.matmul(attention, self.memory_value)
        output = self.output_norm(sequence + self.dropout(recalled))
        return output, attention


class GEANet(nn.Module):
    """
    Corrected GEA-Net.

    Input shape is [batch, sensors, time] or
    [batch, sensors, time, input_features].
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        input_dim: int = 1,
        embed_dim: int = 16,
        memory_size: int = 16,
        graph_layers: int = 2,
        graph_beta: float = 0.5,
        temporal_kernel_size: int = 7,
        dropout: float = 0.1,
        attention_normalization_steps: int = 1,
    ) -> None:
        super().__init__()
        if num_sensors < 1 or num_classes < 2:
            raise ValueError("num_sensors >= 1 and num_classes >= 2 are required")
        if graph_layers < 1:
            raise ValueError("graph_layers must be positive")
        if graph_beta < 0:
            raise ValueError("graph_beta must be non-negative")

        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.graph_layers = graph_layers
        self.graph_beta = graph_beta

        self.temporal_stem = TemporalStem(
            input_dim=input_dim,
            embed_dim=embed_dim,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.external_attention = ExternalAttention(
            embed_dim=embed_dim,
            memory_size=memory_size,
            dropout=dropout,
            normalization_steps=attention_normalization_steps,
        )
        self.edge_logits = nn.Parameter(torch.zeros(graph_layers, num_sensors, num_sensors))
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * embed_dim, num_classes)
        self.register_buffer("adjacency_prior", torch.eye(num_sensors))

    @property
    def feature_dim(self) -> int:
        return 2 * self.embed_dim

    def set_adjacency_prior(self, adjacency: Tensor) -> None:
        adjacency = torch.as_tensor(
            adjacency,
            dtype=self.adjacency_prior.dtype,
            device=self.adjacency_prior.device,
        )
        expected = (self.num_sensors, self.num_sensors)
        if tuple(adjacency.shape) != expected:
            raise ValueError(f"adjacency must have shape {expected}, got {tuple(adjacency.shape)}")
        adjacency = 0.5 * (adjacency + adjacency.t())
        adjacency = adjacency.clamp_min(0)
        adjacency.fill_diagonal_(1.0)
        self.adjacency_prior.copy_(adjacency)

    def _sensor_attention_affinity(self, attention: Tensor) -> Tensor:
        # attention: [B, N, T, alpha]. Cosine similarity fixes the severe scale
        # shrinkage of the paper's 1/(T*alpha) dot-product average.
        flat = F.normalize(attention.flatten(start_dim=2), p=2, dim=-1, eps=1e-8)
        affinity = torch.bmm(flat, flat.transpose(1, 2)).clamp(0.0, 1.0)
        return 0.5 * (affinity + affinity.transpose(1, 2))

    def _graph_encode(self, node_features: Tensor, attention_affinity: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = node_features.shape[0]
        identity = torch.eye(
            self.num_sensors,
            dtype=node_features.dtype,
            device=node_features.device,
        ).expand(batch_size, -1, -1)
        prior = self.adjacency_prior.to(dtype=node_features.dtype, device=node_features.device)
        layers = [node_features]
        current = node_features
        gate_values = []

        for layer_index in range(self.graph_layers):
            logits = self.edge_logits[layer_index]
            gate = torch.sigmoid(0.5 * (logits + logits.t()))
            gate_values.append(gate)

            # One learned structural gate only. This removes the duplicated
            # W_e/W_edge gating in equations (2) and (8) of the manuscript.
            structural = prior * gate
            adjacency = structural.unsqueeze(0) + self.graph_beta * attention_affinity
            adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2)) + identity
            normalized = symmetric_normalize(adjacency)
            current = torch.bmm(normalized, current)
            layers.append(current)

        graph_features = torch.stack(layers, dim=0).mean(dim=0)
        gates = torch.stack(gate_values)
        off_diagonal = 1.0 - torch.eye(
            self.num_sensors,
            dtype=gates.dtype,
            device=gates.device,
        )
        active_prior = (prior > 0).to(gates.dtype) * off_diagonal
        denominator = active_prior.sum().clamp_min(1.0) * self.graph_layers
        edge_sparsity = (gates * active_prior).sum() / denominator
        return graph_features, edge_sparsity

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError(f"Expected [B,N,T] or [B,N,T,F], got {tuple(x.shape)}")
        batch_size, num_sensors, time_steps, input_dim = x.shape
        if num_sensors != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {num_sensors}")
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        flattened = x.reshape(batch_size * num_sensors, time_steps, input_dim)
        stemmed = self.temporal_stem(flattened)
        temporal, attention = self.external_attention(stemmed)
        temporal = temporal.reshape(batch_size, num_sensors, time_steps, self.embed_dim)
        attention = attention.reshape(batch_size, num_sensors, time_steps, -1)

        sequence_nodes = temporal.mean(dim=2)
        attention_affinity = self._sensor_attention_affinity(attention)
        graph_nodes, edge_sparsity = self._graph_encode(sequence_nodes, attention_affinity)

        h_sequence = sequence_nodes.mean(dim=1)
        h_graph = graph_nodes.mean(dim=1)
        fused = self.fusion_dropout(torch.cat([h_sequence, h_graph], dim=-1))
        consistency = (
            F.normalize(h_sequence, dim=-1) - F.normalize(h_graph, dim=-1)
        ).pow(2).sum(dim=-1)
        auxiliary = {
            "h_sequence": h_sequence,
            "h_graph": h_graph,
            "attention_affinity": attention_affinity,
            "consistency": consistency,
            "edge_sparsity": edge_sparsity,
        }
        return fused, auxiliary

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
        features_only: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        features, auxiliary = self.forward_features(x)
        output = features if features_only else self.classifier(features)
        if return_aux:
            return output, auxiliary
        return output

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "num_sensors": self.num_sensors,
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
            "memory_size": self.external_attention.memory_size,
            "graph_layers": self.graph_layers,
            "graph_beta": self.graph_beta,
            "temporal_kernel_size": self.temporal_stem.conv.kernel_size[0],
            "dropout": self.fusion_dropout.p,
            "attention_normalization_steps": self.external_attention.normalization_steps,
        }
