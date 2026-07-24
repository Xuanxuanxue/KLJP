"""Static confusion-label graphs and a small pure-PyTorch graph encoder.

The graph is deliberately built once from training labels and label-definition
embeddings.  It is not rebuilt per batch or per epoch.  The first version uses
cosine similarity only; WRD can be added to ``semantic_adjacency`` later.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


GRAPH_CACHE_VERSION = 3


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
    if cooccurrence.is_sparse:
        raise TypeError("cooccurrence must be a dense tensor")
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


def keep_topk_neighbors(adj: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep at most ``k`` non-zero off-diagonal neighbors per node.

    The graph is symmetrised after pruning.  Zero-valued entries selected by
    ``topk`` are explicitly discarded, so a node with fewer than ``k`` real
    neighbors does not acquire fake edges.
    """

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {tuple(adj.shape)}")
    if adj.is_sparse:
        raise TypeError("confusion graph adjacency must be a dense tensor")
    if k is None:
        result = adj.clone()
        result.fill_diagonal_(0)
        return result
    if int(k) < 0:
        raise ValueError(f"top-k must be non-negative, got {k}")

    n = adj.size(0)
    work = adj.detach().to(dtype=torch.float32).clone()
    work.fill_diagonal_(0)
    if n == 0 or k == 0:
        return torch.zeros_like(adj)
    keep = min(int(k), n - 1)
    values, indices = torch.topk(work, k=keep, dim=1)
    values = values.masked_fill(values <= 0, 0)
    new_adj = torch.zeros_like(work)
    new_adj.scatter_(1, indices, values)
    new_adj = torch.maximum(new_adj, new_adj.T)
    if adj.dtype == torch.bool:
        return new_adj.gt(0)
    return new_adj.to(dtype=adj.dtype)


def normalized_adjacency(
    adjacency: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "graph",
) -> torch.Tensor:
    """Return ``D^(-1/2) (A + I) D^(-1/2)`` with explicit safety checks."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{name} adjacency must be square, got {tuple(adjacency.shape)}")
    if adjacency.is_sparse:
        raise TypeError(f"{name} adjacency must be a dense tensor")
    n = adjacency.size(0)
    work_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    adj = adjacency.to(device=device, dtype=work_dtype)
    if not torch.isfinite(adj).all():
        raise FloatingPointError(f"{name} adjacency contains nan or inf")
    identity = torch.eye(n, device=device, dtype=work_dtype)
    adj_hat = adj + identity
    if n == 0:
        return adj_hat.to(dtype=dtype)
    degree = adj_hat.sum(dim=1)
    if not torch.isfinite(degree).all():
        raise FloatingPointError(f"{name} degree contains nan or inf")
    if (degree < 1).any():
        raise FloatingPointError(
            f"{name} has degree < 1 after adding self-loops: {degree.min().item()}"
        )
    inv_sqrt_degree = degree.rsqrt()
    normalised = inv_sqrt_degree[:, None] * adj_hat * inv_sqrt_degree[None, :]
    nan_count = int(torch.isnan(normalised).sum().item())
    inf_count = int(torch.isinf(normalised).sum().item())
    if nan_count or inf_count:
        raise FloatingPointError(
            f"{name} normalized adjacency is invalid: nan={nan_count}, inf={inf_count}"
        )
    return normalised.to(dtype=dtype)


def mean_offdiag_cosine(x: torch.Tensor) -> float:
    """Mean pairwise cosine similarity, excluding the diagonal."""

    if x.ndim != 2:
        raise ValueError(f"x must have shape (num_labels, hidden_dim), got {tuple(x.shape)}")
    n = x.size(0)
    if n < 2:
        return 0.0
    x = F.normalize(x.float(), p=2, dim=-1)
    sim = x @ x.T
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return float(sim[mask].mean().item())


def load_or_build_confusion_graph(
    *,
    cache_path: str | Path,
    label_embeddings: torch.Tensor,
    cooccurrence: torch.Tensor,
    threshold: float,
    label_order: Sequence[str],
    topk: int | None = None,
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
            and payload.get("cache_version") == GRAPH_CACHE_VERSION
            and "adjacency" in payload
            and "threshold" in payload
            and "num_nodes" in payload
            and "label_order" in payload
            and payload.get("topk") == (None if topk is None else int(topk))
            and int(payload["num_nodes"]) == num_nodes
            and float(payload["threshold"]) == float(threshold)
            and [str(x) for x in payload["label_order"]] == expected_order
        )
        if valid:
            adjacency = payload["adjacency"].to(dtype=torch.bool, device="cpu")
            if tuple(adjacency.shape) == (num_nodes, num_nodes):
                adjacency = adjacency | adjacency.T
                adjacency.fill_diagonal_(False)
                adjacency = keep_topk_neighbors(adjacency, topk)
                return adjacency, graph_statistics(adjacency), True

    adjacency = confusion_adjacency(label_embeddings, cooccurrence, threshold)
    adjacency = keep_topk_neighbors(adjacency, topk)
    payload = {
        "cache_version": GRAPH_CACHE_VERSION,
        "adjacency": adjacency,
        "threshold": float(threshold),
        "num_nodes": num_nodes,
        "label_order": expected_order,
        "topk": None if topk is None else int(topk),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return adjacency, graph_statistics(adjacency), False


class ConfusionGraphEncoder(nn.Module):
    """One-layer pure-PyTorch masked multi-head graph-attention encoder.

    Input and output have shape ``(batch, num_labels, hidden_dim)``.  Attention
    is retained only for confusion edges and temporary self-loops.  The saved
    graph remains diagonal-free for faithful graph statistics and cache
    validation.  ``residual_weight`` is the task-specific lambda_g in
    ``LayerNorm(P + lambda_g * GAT(P))``.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        weight: float = 0.10,
        dropout: float = 0.10,
        name: str = "graph",
    ) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError(f"graph heads must be positive, got {heads}")
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by graph heads ({heads})"
            )
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.weight = float(weight)
        self.name = str(name)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._diagnostics_logged = False

    def forward(
        self,
        label_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        residual_weight: float = 1.0,
    ) -> torch.Tensor:
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

        adjacency = adjacency.to(
            device=label_embeddings.device, dtype=torch.bool
        )
        if not torch.equal(adjacency, adjacency.T):
            raise ValueError(f"{self.name} adjacency must be symmetric")
        if torch.diag(adjacency).any():
            raise ValueError(
                f"{self.name} stored adjacency must not contain self-loops"
            )

        # Add self-loops only to the attention mask.  The registered/cached
        # B_cr matrix itself remains diagonal-free.
        self_loops = torch.eye(
            num_labels, device=label_embeddings.device, dtype=torch.bool
        )
        attention_mask = (adjacency | self_loops)[None, None, :, :]

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            # (B, N, D) -> (B, heads, N, D_head)
            return values.reshape(
                batch_size, num_labels, self.heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(label_embeddings))
        key = split_heads(self.k_proj(label_embeddings))
        value = split_heads(self.v_proj(label_embeddings))
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~attention_mask, torch.finfo(scores.dtype).min
        )
        # Computing softmax in fp32 avoids underflow in mixed-precision
        # training; self-loops guarantee at least one valid key in every row.
        attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        dropped_attention = self.attention_dropout(attention)
        message = torch.matmul(dropped_attention, value)
        message = message.transpose(1, 2).contiguous().reshape(
            batch_size, num_labels, hidden_dim
        )
        message = self.output_dropout(self.out_proj(message))

        scale = self.weight * float(residual_weight)
        enhanced = self.layer_norm(label_embeddings + scale * message)

        if not self._diagnostics_logged:
            degree = adjacency.sum(dim=1)
            print(f"[{self.name} graph] adj shape:", tuple(adjacency.shape))
            print(f"[{self.name} graph] encoder: masked multi-head GAT")
            print(f"[{self.name} graph] attention heads:", self.heads)
            print(f"[{self.name} graph] degree min:", (degree + 1).min().item())
            print(f"[{self.name} graph] degree max:", (degree + 1).max().item())
            print(f"[{self.name} graph] isolated node count:", (degree == 0).sum().item())
            print(f"[{self.name} graph] attention nan count:", torch.isnan(attention).sum().item())
            print(f"[{self.name} graph] attention inf count:", torch.isinf(attention).sum().item())
            valid_attention = attention.masked_select(
                attention_mask.expand_as(attention)
            )
            masked_attention = attention.masked_select(
                (~attention_mask).expand_as(attention)
            )
            row_sum_error = (attention.sum(dim=-1) - 1.0).abs().max()
            print(
                f"[{self.name} graph] attention valid min:",
                0.0 if valid_attention.numel() == 0 else valid_attention.min().item(),
            )
            print(
                f"[{self.name} graph] masked attention max:",
                0.0 if masked_attention.numel() == 0 else masked_attention.max().item(),
            )
            print(f"[{self.name} graph] attention row-sum max error:", row_sum_error.item())
            self._diagnostics_logged = True
        return enhanced

    @staticmethod
    def smoothness_loss(label_embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Optional mean squared difference over real (diagonal-free) edges."""

        edges = torch.triu(adjacency.to(device=label_embeddings.device, dtype=torch.bool), diagonal=1)
        row, col = edges.nonzero(as_tuple=True)
        if row.numel() == 0:
            return label_embeddings.sum() * 0.0
        difference = label_embeddings[:, row, :] - label_embeddings[:, col, :]
        return difference.square().mean()


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
        article_alpha: float = 0.0,
        charge_alpha: float = 0.0,
        topk: int | None = None,
    ) -> None:
        """Build/load each requested graph once during model construction."""

        self.use_article_confusion_graph = bool(use_article)
        self.use_charge_confusion_graph = bool(use_charge)
        self.article_graph_alpha = float(article_alpha)
        self.charge_graph_alpha = float(charge_alpha)
        if self.article_graph_alpha < 0 or self.charge_graph_alpha < 0:
            raise ValueError("confusion graph alpha must be non-negative")
        self.confusion_graph_stats: dict[str, dict[str, float | int]] = {}
        self.confusion_graph_loss = torch.zeros(())
        self.confusion_graph_diagnostics: dict[str, dict[str, float]] = {}
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
                topk=topk,
            )
            self.register_buffer("article_confusion_adj", adjacency)
            self.article_confusion_encoder = ConfusionGraphEncoder(
                self.hid_dim, heads=heads, weight=weight, dropout=dropout, name="article"
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
                topk=topk,
            )
            self.register_buffer("charge_confusion_adj", adjacency)
            self.charge_confusion_encoder = ConfusionGraphEncoder(
                self.hid_dim, heads=heads, weight=weight, dropout=dropout, name="charge"
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
            encoded = encoded.masked_fill(
                (~attention_mask).unsqueeze(-1), torch.finfo(encoded.dtype).min
            )
            encoded = torch.max(encoded, dim=1).values
            if missing_indices.numel() > 0:
                encoded = encoded.index_copy(0, missing_indices, fill_values)
            if was_training:
                self.transformer_enc.train()
            return encoded

        return self.label_emb_char.weight if char_flag else self.label_emb_art.weight

    def apply_confusion_graphs(
        self, article_label_emb: torch.Tensor, charge_label_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply independent residual GAT enhancement immediately before decoders."""

        original_article = article_label_emb
        original_charge = charge_label_emb
        graph_losses = []
        diagnostics = {}

        if getattr(self, "use_article_confusion_graph", False):
            if self.article_graph_alpha != 0.0:
                article_label_emb = self.article_confusion_encoder(
                    original_article,
                    self.article_confusion_adj,
                    residual_weight=self.article_graph_alpha,
                )
            graph_losses.append(
                self.article_confusion_encoder.smoothness_loss(
                    original_article, self.article_confusion_adj
                )
            )
            diagnostics["article"] = self._embedding_diagnostics(
                original_article[0], article_label_emb[0]
            )
        if getattr(self, "use_charge_confusion_graph", False):
            if self.charge_graph_alpha != 0.0:
                charge_label_emb = self.charge_confusion_encoder(
                    original_charge,
                    self.charge_confusion_adj,
                    residual_weight=self.charge_graph_alpha,
                )
            graph_losses.append(
                self.charge_confusion_encoder.smoothness_loss(
                    original_charge, self.charge_confusion_adj
                )
            )
            diagnostics["charge"] = self._embedding_diagnostics(
                original_charge[0], charge_label_emb[0]
            )
        self.confusion_graph_loss = (
            torch.stack(graph_losses).mean()
            if graph_losses
            else original_article.sum() * 0.0
        )
        self.confusion_graph_diagnostics = diagnostics
        if self.article_graph_alpha == 0.0:
            assert torch.allclose(article_label_emb, original_article, atol=1e-6)
        if self.charge_graph_alpha == 0.0:
            assert torch.allclose(charge_label_emb, original_charge, atol=1e-6)
        return article_label_emb, charge_label_emb

    @staticmethod
    def _embedding_diagnostics(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
        return {
            "cosine_before": mean_offdiag_cosine(before),
            "cosine_after": mean_offdiag_cosine(after),
            "variance_before": float(before.var(dim=0).mean().item()),
            "variance_after": float(after.var(dim=0).mean().item()),
        }
