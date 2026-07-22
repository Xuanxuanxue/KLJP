"""Static confusion-label graphs and a small pure-PyTorch graph encoder.

The graph is deliberately built once from training labels and label-definition
embeddings.  It is not rebuilt per batch or per epoch.  The first version uses
cosine similarity only; WRD can be added to ``semantic_adjacency`` later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_order_from_mapping(label2idx: Mapping[str, int]) -> list[str]:
    """Return the exact model order without changing the original mapping."""

    ordered = sorted(label2idx.items(), key=lambda item: int(item[1]))
    ids = [int(item[1]) for item in ordered]
    if ids != list(range(len(ids))):
        raise ValueError("label ids must be contiguous and start at zero")
    return [str(item[0]) for item in ordered]


def build_training_cooccurrence(
    filename: str | Path,
    label2idx: Mapping[str, int],
    label_kind: str,
) -> torch.Tensor:
    """Build ``B_cc`` from *only* the JSONL training file.

    ``label_kind`` is ``"article"`` or ``"charge"``.  Validation and test
    files are intentionally not accepted by the training caller.
    """

    if label_kind not in {"article", "charge"}:
        raise ValueError(f"unsupported label kind: {label_kind}")
    num_nodes = len(label2idx)
    cooccurrence = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    meta_key = "relevant_articles" if label_kind == "article" else "accusation"

    with Path(filename).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_labels = record["meta"][meta_key]
            labels = []
            for raw_label in raw_labels:
                key = str(raw_label) if label_kind == "article" else raw_label
                if key not in label2idx:
                    raise KeyError(
                        f"{label_kind} label {key!r} at line {line_number} "
                        "is missing from the static mapping"
                    )
                labels.append(int(label2idx[key]))
            labels = sorted(set(labels))
            if labels:
                index = torch.tensor(labels, dtype=torch.long)
                cooccurrence[index[:, None], index[None, :]] = True

    # B_cc is a relation matrix and is kept symmetric even if input handling
    # changes in the future.  Its diagonal is harmless because B_cr resets it.
    return cooccurrence | cooccurrence.T


def semantic_adjacency(label_embeddings: torch.Tensor, threshold: float) -> torch.Tensor:
    """Build the symmetric semantic relation matrix using cosine similarity.

    WRD is intentionally not implemented in this first version; cosine
    similarity is the only semantic relation so that this module stays light.
    """

    if label_embeddings.ndim != 2:
        raise ValueError(
            "label_embeddings must have shape (num_labels, hidden_dim), "
            f"got {tuple(label_embeddings.shape)}"
        )
    embeddings = F.normalize(label_embeddings.detach().float(), p=2, dim=-1)
    similarity = embeddings @ embeddings.T
    relation = similarity.ge(float(threshold))
    relation = relation | relation.T
    relation.fill_diagonal_(False)
    return relation


def confusion_adjacency(
    label_embeddings: torch.Tensor,
    cooccurrence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return ``B_cr = B_sr AND (1 - B_cc)`` with the required invariants."""

    num_nodes = label_embeddings.shape[0]
    expected_shape = (num_nodes, num_nodes)
    if tuple(cooccurrence.shape) != expected_shape:
        raise ValueError(
            f"cooccurrence must have shape {expected_shape}, got {tuple(cooccurrence.shape)}"
        )
    cooccurrence = cooccurrence.to(dtype=torch.bool, device="cpu")
    cooccurrence = cooccurrence | cooccurrence.T
    semantic = semantic_adjacency(label_embeddings.cpu(), threshold)
    adjacency = semantic & ~cooccurrence
    adjacency.fill_diagonal_(False)
    return adjacency | adjacency.T


def graph_statistics(adjacency: torch.Tensor) -> dict[str, float | int]:
    """Return undirected-edge statistics for a diagonal-free adjacency matrix."""

    adjacency = adjacency.to(dtype=torch.bool)
    num_nodes = int(adjacency.shape[0])
    edge_count = int(torch.triu(adjacency, diagonal=1).sum().item())
    degrees = adjacency.sum(dim=1)
    return {
        "node_count": num_nodes,
        "edge_count": edge_count,
        "average_degree": (2.0 * edge_count / num_nodes) if num_nodes else 0.0,
        "isolated_node_count": int(degrees.eq(0).sum().item()),
    }


def load_or_build_confusion_graph(
    *,
    cache_path: str | Path,
    label_embeddings: torch.Tensor,
    cooccurrence: torch.Tensor,
    threshold: float,
    label_order: Sequence[str],
) -> tuple[torch.Tensor, dict[str, float | int], bool]:
    """Load a validated graph cache or build and cache it once.

    The boolean return value indicates whether the cache was used.
    """

    cache_path = Path(cache_path)
    num_nodes = int(label_embeddings.shape[0])
    expected_order = [str(label) for label in label_order]
    if len(expected_order) != num_nodes:
        raise ValueError("label_order length must match label embedding count")

    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        valid = (
            isinstance(payload, dict)
            and "adjacency" in payload
            and "threshold" in payload
            and "num_nodes" in payload
            and "label_order" in payload
            and int(payload["num_nodes"]) == num_nodes
            and float(payload["threshold"]) == float(threshold)
            and [str(x) for x in payload["label_order"]] == expected_order
        )
        if valid:
            adjacency = payload["adjacency"].to(dtype=torch.bool, device="cpu")
            if tuple(adjacency.shape) == (num_nodes, num_nodes):
                adjacency = adjacency | adjacency.T
                adjacency.fill_diagonal_(False)
                return adjacency, graph_statistics(adjacency), True

    adjacency = confusion_adjacency(label_embeddings, cooccurrence, threshold)
    payload = {
        "adjacency": adjacency,
        "threshold": float(threshold),
        "num_nodes": num_nodes,
        "label_order": expected_order,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return adjacency, graph_statistics(adjacency), False


class ConfusionGraphEncoder(nn.Module):
    """Small multi-head masked graph-attention residual encoder.

    Input and output have shape ``(batch, num_labels, hidden_dim)``.  A
    self-loop is added only inside the forward pass; the saved graph remains
    diagonal-free for faithful graph statistics and cache validation.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        weight: float = 0.10,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by graph heads ({heads})"
            )
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.weight = float(weight)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, label_embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if label_embeddings.ndim != 3:
            raise ValueError(
                "label_embeddings must have shape (batch, num_labels, hidden_dim), "
                f"got {tuple(label_embeddings.shape)}"
            )
        batch_size, num_labels, hidden_dim = label_embeddings.shape
        if hidden_dim != self.hidden_dim or tuple(adjacency.shape) != (num_labels, num_labels):
            raise ValueError(
                "graph input shape mismatch: "
                f"embeddings={tuple(label_embeddings.shape)}, adjacency={tuple(adjacency.shape)}"
            )

        adjacency = adjacency.to(device=label_embeddings.device, dtype=torch.bool)
        self_loop = torch.eye(num_labels, device=label_embeddings.device, dtype=torch.bool)
        attention_mask = (adjacency | self_loop)[None, None, :, :]

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(batch_size, num_labels, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.q_proj(label_embeddings))
        key = split_heads(self.k_proj(label_embeddings))
        value = split_heads(self.v_proj(label_embeddings))
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        message = torch.matmul(attention, value)
        message = message.transpose(1, 2).contiguous().view(batch_size, num_labels, hidden_dim)
        message = self.out_proj(message)
        return self.layer_norm(label_embeddings + self.weight * message)


class ConfusionGraphMixin:
    """Shared model wiring for Al_Trans and CNN_Trans."""

    def setup_confusion_graphs(
        self,
        *,
        maps: Mapping[str, Mapping[str, int]],
        article_cooccurrence: torch.Tensor | None,
        charge_cooccurrence: torch.Tensor | None,
        cache_dir: str | Path,
        use_article: bool,
        use_charge: bool,
        threshold: float,
        weight: float,
        heads: int,
        dropout: float,
    ) -> None:
        """Build/load each requested graph once during model construction."""

        self.use_article_confusion_graph = bool(use_article)
        self.use_charge_confusion_graph = bool(use_charge)
        self.confusion_graph_stats: dict[str, dict[str, float | int]] = {}
        if not (self.use_article_confusion_graph or self.use_charge_confusion_graph):
            return

        if self.emb_dim != self.hid_dim:
            raise ValueError(
                "confusion graph requires label embedding dimension == decoder dimension; "
                f"got emb_dim={self.emb_dim}, hid_dim={self.hid_dim}"
            )

        # The graph uses the same initial label-definition encoder as the
        # model.  If these parameters are trainable, the graph remains static
        # after this one-time construction, as required by the experiment.
        with torch.no_grad():
            article_embeddings = self._initial_label_embeddings(char_flag=False)
            charge_embeddings = self._initial_label_embeddings(char_flag=True)

        cache_dir = Path(cache_dir)
        if self.use_article_confusion_graph:
            if article_cooccurrence is None:
                raise ValueError("article training co-occurrence is required when its graph is enabled")
            article_order = label_order_from_mapping(maps["article2idx"])
            adjacency, stats, _ = load_or_build_confusion_graph(
                cache_path=cache_dir / "article_confusion_graph.pt",
                label_embeddings=article_embeddings,
                cooccurrence=article_cooccurrence,
                threshold=threshold,
                label_order=article_order,
            )
            self.register_buffer("article_confusion_adj", adjacency)
            self.article_confusion_encoder = ConfusionGraphEncoder(
                self.hid_dim, heads=heads, weight=weight, dropout=dropout
            )
            self.confusion_graph_stats["article"] = stats

        if self.use_charge_confusion_graph:
            if charge_cooccurrence is None:
                raise ValueError("charge training co-occurrence is required when its graph is enabled")
            charge_order = label_order_from_mapping(maps["charge2idx"])
            adjacency, stats, _ = load_or_build_confusion_graph(
                cache_path=cache_dir / "charge_confusion_graph.pt",
                label_embeddings=charge_embeddings,
                cooccurrence=charge_cooccurrence,
                threshold=threshold,
                label_order=charge_order,
            )
            self.register_buffer("charge_confusion_adj", adjacency)
            self.charge_confusion_encoder = ConfusionGraphEncoder(
                self.hid_dim, heads=heads, weight=weight, dropout=dropout
            )
            self.confusion_graph_stats["charge"] = stats

    def _initial_label_embeddings(self, char_flag: bool) -> torch.Tensor:
        """Encode definition rows once, returning ``(num_labels, hidden_dim)``."""

        if hasattr(self, "transformer_enc"):
            if char_flag:
                details = self.charge_details
                attention_mask = self.charge_detail_attention_mask
                missing_indices = self.charge_missing_indices
                fill_values = self.fill_char
            else:
                details = self.article_details
                attention_mask = self.article_detail_attention_mask
                missing_indices = self.article_missing_indices
                fill_values = self.fill_article
            was_training = self.transformer_enc.training
            self.transformer_enc.eval()
            encoded = self.embedding(details)
            encoded = encoded.masked_fill((~attention_mask).unsqueeze(-1), 0.0)
            encoded = self.transformer_enc(
                encoded, src_key_padding_mask=~attention_mask.to(dtype=torch.bool)
            )
            encoded = encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
            if missing_indices.numel() > 0:
                encoded = encoded.index_copy(0, missing_indices, fill_values)
            if was_training:
                self.transformer_enc.train()
            return encoded

        return self.label_emb_char.weight if char_flag else self.label_emb_art.weight

    def apply_confusion_graphs(
        self, article_label_emb: torch.Tensor, charge_label_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhance labels immediately before the two Transformer Decoders."""

        if getattr(self, "use_article_confusion_graph", False):
            article_label_emb = self.article_confusion_encoder(
                article_label_emb, self.article_confusion_adj
            )
        if getattr(self, "use_charge_confusion_graph", False):
            charge_label_emb = self.charge_confusion_encoder(
                charge_label_emb, self.charge_confusion_adj
            )
        return article_label_emb, charge_label_emb
