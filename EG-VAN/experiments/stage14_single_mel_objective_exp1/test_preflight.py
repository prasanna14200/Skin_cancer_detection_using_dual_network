"""CPU-only checks; never builds a HAM loader or starts training."""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as runner


class FakeScaler:
    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)

    def get_scale(self):
        return self.state["scale"]


def fixture():
    cfg, rule, _ = runner.read_preregistration()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    for epoch in range(1, 9):
        scheduler.step(0.1 / epoch)
    history = [{"epoch": epoch, "train_loss": 0.2, "val_loss": 0.1,
        "val_accuracy": 0.80, "val_macro_f1": 0.65,
        "val_mel_recall": 62 / 107, "val_mel_f1": 0.55,
        "val_nv_recall": 0.92, "eligible": False} for epoch in range(1, 9)]
    generator = torch.Generator().manual_seed(42)
    state = {"experiment": runner.NAME, "epoch": 8, "configuration": cfg,
        "config_sha256": runner.CONFIG_SHA, "selection_rule_sha256": runner.RULE_SHA,
        "numerical_protocol_sha256": runner.NUMERICAL_SHA,
        "runner_sha256": runner.base.sha256(Path(runner.__file__)),
        "class_order": list(runner.base.CLASSES), "architecture": cfg["architecture"],
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": copy.deepcopy(scheduler.state_dict()),
        "scaler_state": {"scale": 4096.0, "growth_factor": 2.0,
            "backoff_factor": 0.5, "growth_interval": 2000, "_growth_tracker": 8},
        "history": history, "best_epoch": None, "best_validation_loss": math.inf,
        "numerical_events": [], "sampler_generator_state": generator.get_state().clone(),
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": [torch.get_rng_state().clone()]}
    return state, cfg, rule


class Stage14PreflightTests(unittest.TestCase):
    def test_single_factor_and_frozen_files(self):
        cfg, rule, numerical = runner.read_preregistration()
        self.assertEqual(cfg["loss"]["mel_multiplier"], 1.5630495442733532)
        self.assertEqual(cfg["sampler"], runner.base.variant_config("A")["sampler"])
        self.assertEqual(numerical["initial_grad_scaler_scale"], 8192.0)
        self.assertEqual(rule["eligible_if_all"]["melanoma_correct_minimum"], 63)

    def test_eligibility_boundary_and_preservation_floors(self):
        _, _, rule = fixture()
        floor = rule["eligible_if_all"]
        metrics = {"accuracy": floor["accuracy_minimum"],
                   "macro_f1": floor["macro_f1_minimum"],
                   "per_class": {"mel": {"support": 107, "recall": 63 / 107,
                                         "f1": floor["melanoma_f1_minimum"]},
                                 "nv": {"recall": floor["nevus_recall_minimum"]}}}
        self.assertTrue(runner.eligible(metrics, rule))
        bad = copy.deepcopy(metrics)
        bad["per_class"]["mel"]["recall"] = 62 / 107
        self.assertFalse(runner.eligible(bad, rule))
        for field in ("accuracy", "macro_f1"):
            bad = copy.deepcopy(metrics)
            bad[field] -= 1e-4
            self.assertFalse(runner.eligible(bad, rule))
        for key, field in (("mel", "f1"), ("nv", "recall")):
            bad = copy.deepcopy(metrics)
            bad["per_class"][key][field] -= 1e-4
            self.assertFalse(runner.eligible(bad, rule))

    def test_valid_epoch8_and_mismatch_rejections(self):
        state, cfg, rule = fixture()
        self.assertEqual(runner.validate_resume(state, cfg, rule)["start_epoch"], 9)
        bad = copy.deepcopy(state)
        bad["configuration"]["loss"]["mel_multiplier"] = 1.0
        with self.assertRaisesRegex(ValueError, "provenance"):
            runner.validate_resume(bad, cfg, rule)
        bad = copy.deepcopy(state)
        bad["history"].pop()
        with self.assertRaisesRegex(ValueError, "epoch/history"):
            runner.validate_resume(bad, cfg, rule)
        bad = copy.deepcopy(state)
        next(iter(bad["model_state"].values())).flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            runner.validate_resume(bad, cfg, rule)
        bad = copy.deepcopy(state)
        next(iter(bad["optimizer_state"]["state"].values()))["exp_inf"].flatten()[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            runner.validate_resume(bad, cfg, rule)
        bad = copy.deepcopy(state)
        bad["scaler_state"]["scale"] = float("nan")
        with self.assertRaisesRegex(ValueError, "GradScaler"):
            runner.validate_resume(bad, cfg, rule)

    def test_rng_and_scaler_restore(self):
        state, cfg, rule = fixture()
        runner.validate_resume(state, cfg, rule)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        scaler, generator, captured_cuda = FakeScaler(), torch.Generator().manual_seed(999), []
        prior = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        try:
            restored = runner.restore_resume(state, model, optimizer, scheduler, scaler,
                generator, cuda_rng_setter=lambda values: captured_cuda.extend(values))
            self.assertEqual(restored[0], 9)
            self.assertEqual(scaler.get_scale(), 4096.0)
            self.assertEqual(random.getstate(), state["python_rng_state"])
            self.assertTrue(np.array_equal(np.random.get_state()[1], state["numpy_rng_state"][1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
            self.assertTrue(torch.equal(generator.get_state(), state["sampler_generator_state"]))
            self.assertTrue(torch.equal(captured_cuda[0], state["cuda_rng_states"][0]))
        finally:
            random.setstate(prior[0]); np.random.set_state(prior[1]); torch.set_rng_state(prior[2])

    def test_missing_checkpoint_no_fresh_fallback(self):
        state, cfg, rule = fixture()
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(runner, "OUT", Path(temp) / "run"):
                with self.assertRaisesRegex(FileNotFoundError, "no fresh fallback"):
                    runner.execute(runner.ROOT, cfg, rule, {}, resume=True)
                self.assertFalse(runner.OUT.exists())

    def test_csv_prefix_and_best_preservation(self):
        state, cfg, rule = fixture()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            out.mkdir()
            for filename in ("config.json", "selection_rule.json", "numerical_protocol.json"):
                (out / filename).write_bytes((runner.HERE / filename).read_bytes())
            with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(state["history"][0]))
                writer.writeheader(); writer.writerows(state["history"][:7])
            (out / "numerical_events.json").write_text("[]", encoding="utf-8")
            with patch.object(runner, "OUT", out):
                runner.verify_run_files_for_resume(state)  # CSV may lag atomic checkpoint by one write.
                state["history"][2].update({"eligible": True, "val_mel_recall": 63 / 107,
                    "val_mel_f1": rule["eligible_if_all"]["melanoma_f1_minimum"],
                    "val_macro_f1": rule["eligible_if_all"]["macro_f1_minimum"],
                    "val_accuracy": rule["eligible_if_all"]["accuracy_minimum"],
                    "val_nv_recall": rule["eligible_if_all"]["nevus_recall_minimum"]})
                state["best_epoch"], state["best_validation_loss"] = 3, 0.1
                best = copy.deepcopy(state)
                best["epoch"], best["history"] = 3, copy.deepcopy(state["history"][:3])
                torch.save(best, out / "best_checkpoint.pt")
                before = runner.base.sha256(out / "best_checkpoint.pt")
                runner.validate_resume(state, cfg, rule)
                self.assertEqual(before, runner.base.sha256(out / "best_checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
