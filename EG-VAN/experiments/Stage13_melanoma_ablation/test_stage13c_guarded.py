"""CPU-safe checks for the proposed Stage 13C numerical guard policy."""
import math
import unittest
from unittest.mock import patch

import torch

import stage13c_guarded as guarded
import train_ablation as base


class FakeScaler:
    def __init__(self, scale):
        self.scale_value = scale

    def get_scale(self):
        return self.scale_value


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_science_and_numerical_policy(self):
        self.assertEqual(guarded.INITIAL_SCALE, 16384.0)
        self.assertEqual(guarded.GATE_BATCHES, 8)
        a, b, c = (base.variant_config(x) for x in "ABC")
        self.assertEqual(a["loss"]["mel_multiplier"], 1.0)
        self.assertEqual(b["loss"]["mel_multiplier"], (5366 / 899) ** 0.25)
        self.assertEqual(c["loss"], a["loss"])
        self.assertEqual(a["sampler"], b["sampler"])
        self.assertEqual(c["sampler"]["mel_weight"], (5366 / 899) ** 0.5)

    def test_guarded_focal_terms_match_frozen_loss(self):
        logits = torch.tensor([[0.2, -0.1, 0.3, 0.5, 1.0, 0.4, -0.2],
                               [1.2, 0.7, -0.1, 0.0, -0.3, 0.2, -0.5]])
        targets = torch.tensor([4, 0])
        for variant in "ABC":
            multiplier = base.variant_config(variant)["loss"]["mel_multiplier"]
            torch.testing.assert_close(guarded.focal_terms(logits, targets, multiplier).mean(),
                                       base.loss(logits, targets, multiplier))

    def test_nonfinite_loss_and_scale_abort(self):
        with self.assertRaises(guarded.NumericalFailure) as raised:
            guarded.require_finite(torch.tensor([1.0, float("nan")]),
                                   {"variant": "B", "epoch": 1, "batch_index": 1}, "per_example_loss")
        self.assertEqual(raised.exception.record["affected_tensor"], "per_example_loss")
        self.assertEqual(raised.exception.record["nan_count"], 1)
        with self.assertRaises(guarded.NumericalFailure):
            guarded.require_valid_scale(FakeScaler(0.0), {"variant": "A", "epoch": 1})
        with self.assertRaises(guarded.NumericalFailure):
            guarded.require_finite_epoch_losses(math.nan, 0.1, "B", 1)

    def test_bad_state_cannot_reach_checkpoint_writer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
        value = model(torch.ones(1, 2)).sum()
        value.backward(); optimizer.step()
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        state = {"scheduler_state": {"best": 0.1, "_last_lr": [0.001]},
                 "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}]}
        with patch.object(base, "save_checkpoint") as save:
            with self.assertRaises(guarded.NumericalFailure):
                guarded.checked_checkpoint(None, state, model, optimizer, FakeScaler(16384), "B", 1)
            save.assert_not_called()

    def test_optimizer_state_nan_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        first_state = next(iter(optimizer.state.values()))
        first_state["exp_avg"].view(-1)[0] = float("nan")
        with self.assertRaises(guarded.NumericalFailure) as raised:
            guarded.require_finite_states(model, optimizer, {"variant": "C", "epoch": 1})
        self.assertEqual(raised.exception.record["affected_tensor"], "optimizer_state/exp_avg")


if __name__ == "__main__":
    unittest.main()
