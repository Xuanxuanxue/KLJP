import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


CONTRA_ROOT = Path(__file__).resolve().parents[1] / "contra"
sys.path.insert(0, str(CONTRA_ROOT))

from confusion_graph import (  # noqa: E402
    ConfusionGraphEncoder,
    build_training_cooccurrence,
    graph_statistics,
    load_or_build_confusion_graph,
)
import models.el_trans as el_trans  # noqa: E402


class DummyElectra(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ConfusionGraphTest(unittest.TestCase):
    def test_training_cooccurrence_uses_jsonl_labels_and_graph_invariants(self):
        maps = {"a0": 0, "a1": 1, "a2": 2}
        with TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.json"
            train_path.write_text(
                "\n".join(
                    [
                        json.dumps({"meta": {"relevant_articles": ["a0", "a1"], "accusation": []}}),
                        json.dumps({"meta": {"relevant_articles": ["a1", "a2"], "accusation": []}}),
                    ]
                ),
                encoding="utf-8",
            )
            cooccurrence = build_training_cooccurrence(train_path, maps, "article")
            embeddings = torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.98, 0.2]])
            adjacency = (embeddings @ embeddings.T).ge(0.98)
            confusion, stats, from_cache = load_or_build_confusion_graph(
                cache_path=Path(temp_dir) / "article_confusion_graph.pt",
                label_embeddings=embeddings,
                cooccurrence=cooccurrence,
                threshold=0.98,
                label_order=["a0", "a1", "a2"],
            )

            self.assertFalse(from_cache)
            self.assertTrue(cooccurrence[0, 1] and cooccurrence[1, 2])
            self.assertFalse(confusion.any())
            self.assertTrue(torch.equal(confusion, confusion.T))
            self.assertEqual(torch.diag(confusion).sum().item(), 0)
            self.assertEqual(stats, graph_statistics(confusion))
            _, _, from_cache = load_or_build_confusion_graph(
                cache_path=Path(temp_dir) / "article_confusion_graph.pt",
                label_embeddings=embeddings,
                cooccurrence=cooccurrence,
                threshold=0.98,
                label_order=["a0", "a1", "a2"],
            )
            self.assertTrue(from_cache)
            _, _, from_cache = load_or_build_confusion_graph(
                cache_path=Path(temp_dir) / "article_confusion_graph.pt",
                label_embeddings=embeddings,
                cooccurrence=cooccurrence,
                threshold=0.98,
                label_order=["a1", "a0", "a2"],
            )
            self.assertFalse(from_cache)

    def test_encoder_keeps_shape_and_updates_only_graph_neighbors(self):
        torch.manual_seed(1)
        encoder = ConfusionGraphEncoder(8, heads=4, weight=0.1, dropout=0.0)
        labels = torch.randn(2, 3, 8, requires_grad=True)
        adjacency = torch.tensor(
            [[False, True, False], [True, False, False], [False, False, False]]
        )
        output = encoder(labels, adjacency)
        self.assertEqual(output.shape, labels.shape)
        output.square().mean().backward()
        self.assertIsNotNone(labels.grad)
        self.assertTrue(torch.isfinite(labels.grad).all())

    def test_al_trans_graph_forward_backward_and_checkpoint_buffers(self):
        maps = {
            "charge2idx": {"c0": 0, "c1": 1, "c2": 2},
            "article2idx": {"a0": 0, "a1": 1},
        }

        def details(rows):
            ids = torch.tensor(rows, dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": ids.ne(0).long(),
                "detail_present_mask": ids.ne(0).any(dim=1),
            }

        with TemporaryDirectory() as temp_dir, patch.object(
            el_trans.AutoModel, "from_pretrained", return_value=DummyElectra()
        ), patch.object(el_trans, "USE_ARTICLE_CONFUSION_GRAPH", True), patch.object(
            el_trans, "USE_CHARGE_CONFUSION_GRAPH", True
        ):
            model = el_trans.Al_Trans(
                vocab_size=32,
                emb_dim=8,
                hid_dim=8,
                max_length=6,
                maps=maps,
                article_details=details([[1, 2, 0], [3, 4, 0]]),
                charge_details=details([[5, 6, 0], [7, 8, 0], [9, 10, 0]]),
                article_cooccurrence=torch.tensor([[True, False], [False, True]]),
                charge_cooccurrence=torch.eye(3, dtype=torch.bool),
                confusion_graph_cache_dir=temp_dir,
            )
            output = model(
                {
                    "justice": {
                        "input_ids": torch.tensor([[11, 12, 13, 0, 0, 0]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
                    }
                }
            )
            self.assertEqual(output["charge"].shape, (1, 3))
            self.assertEqual(output["article"].shape, (1, 2))
            self.assertEqual(model.article_confusion_adj.shape, (2, 2))
            self.assertEqual(model.charge_confusion_adj.shape, (3, 3))
            self.assertTrue(torch.equal(model.article_confusion_adj, model.article_confusion_adj.T))
            self.assertTrue(torch.equal(model.charge_confusion_adj, model.charge_confusion_adj.T))
            self.assertEqual(torch.diag(model.article_confusion_adj).sum().item(), 0)
            self.assertEqual(torch.diag(model.charge_confusion_adj).sum().item(), 0)
            (output["charge"].mean() + output["article"].mean()).backward()
            self.assertTrue(any("confusion_encoder" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
