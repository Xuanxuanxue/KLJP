import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


CONTRA_ROOT = Path(__file__).resolve().parents[1] / "contra"
sys.path.insert(0, str(CONTRA_ROOT))

import models.el_trans as el_trans  # noqa: E402
from models.cnn_trans import CNN_Encoder  # noqa: E402
from masking import masked_mean_pool  # noqa: E402


class DummyElectra(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def detail_batch(input_ids, attention_mask, present_mask):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "detail_present_mask": torch.tensor(present_mask, dtype=torch.bool),
    }


class EncoderDecoderTest(unittest.TestCase):
    def build_model(self):
        maps = {
            "charge2idx": {"c0": 0, "c1": 1, "c2": 2},
            "article2idx": {"a0": 0, "a1": 1},
        }
        article_details = detail_batch(
            [[1, 2, 0, 0], [3, 4, 5, 0]],
            [[1, 1, 0, 0], [1, 1, 1, 0]],
            [True, True],
        )
        charge_details = detail_batch(
            [[6, 7, 0, 0], [0, 0, 0, 0], [8, 9, 10, 0]],
            [[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]],
            [True, False, True],
        )
        with patch.object(
            el_trans.AutoModel,
            "from_pretrained",
            return_value=DummyElectra(),
        ):
            return el_trans.Al_Trans(
                vocab_size=32,
                emb_dim=8,
                hid_dim=8,
                max_length=6,
                maps=maps,
                article_details=article_details,
                charge_details=charge_details,
            )

    def test_forward_backward_with_two_views(self):
        model = self.build_model()
        self.assertTrue(model.fc_charge.bias)
        self.assertTrue(model.fc_article.bias)
        self.assertIsNotNone(model.fc_charge.b)
        self.assertIsNotNone(model.fc_article.b)
        view = torch.tensor([[11, 12, 13, 0, 0, 0]])
        data = {
            "justice": {
                "input_ids": [view, view.clone()],
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]),
            }
        }

        output = model(data)
        self.assertEqual(output["charge"].shape, (2, 3))
        self.assertEqual(output["article"].shape, (2, 2))
        self.assertEqual(model.charge_missing_indices.tolist(), [1])

        loss = output["charge"].square().mean() + output["article"].square().mean()
        loss.backward()
        self.assertIsNotNone(model.fill_char.grad)
        self.assertTrue(torch.isfinite(model.fill_char.grad).all())
        self.assertTrue(torch.all(output["charge"].ge(0)))
        self.assertTrue(torch.all(output["charge"].le(1)))
        self.assertTrue(torch.all(output["article"].ge(0)))
        self.assertTrue(torch.all(output["article"].le(1)))

    def test_padding_token_values_do_not_change_logits(self):
        model = self.build_model().eval()
        input_ids = torch.tensor([[11, 12, 13, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

        with torch.no_grad():
            baseline = model(
                {"justice": {"input_ids": input_ids, "attention_mask": attention_mask}}
            )

            padded_input_ids = input_ids.clone()
            padded_input_ids[attention_mask.eq(0)] = 31
            model.article_details[~model.article_detail_attention_mask] = 30
            model.charge_details[~model.charge_detail_attention_mask] = 29
            changed_padding = model(
                {
                    "justice": {
                        "input_ids": padded_input_ids,
                        "attention_mask": attention_mask,
                    }
                }
            )

        torch.testing.assert_close(
            baseline["charge"], changed_padding["charge"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            baseline["article"], changed_padding["article"], atol=1e-6, rtol=1e-6
        )

    def test_masked_mean_pool_ignores_padding_and_handles_empty_rows(self):
        sequence = torch.tensor(
            [[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]],
             [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
        )
        valid_mask = torch.tensor([[True, True, False], [False, False, False]])

        pooled = masked_mean_pool(sequence, valid_mask)

        torch.testing.assert_close(pooled[0], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(pooled[1], torch.zeros(2))

    def test_fully_masked_fact_is_rejected(self):
        model = self.build_model().eval()
        with self.assertRaisesRegex(ValueError, "fully masked samples"):
            model(
                {
                    "justice": {
                        "input_ids": torch.zeros((1, 6), dtype=torch.long),
                        "attention_mask": torch.zeros((1, 6), dtype=torch.long),
                    }
                }
            )

    def test_cnn_mask_prevents_padding_values_from_leaking(self):
        encoder = CNN_Encoder(hid_dim=4, dropout=0.0, num_layers=2).eval()
        padding_mask = torch.tensor([[False, False, True, True]])
        baseline = torch.randn(1, 4, 4)
        changed = baseline.clone()
        changed[:, 2:] = 1000.0

        with torch.no_grad():
            baseline_output = encoder(baseline, padding_mask)
            changed_output = encoder(changed, padding_mask)

        torch.testing.assert_close(baseline_output, changed_output)
        self.assertTrue(changed_output[:, 2:].eq(0).all())


if __name__ == "__main__":
    unittest.main()
