"""CPU-only, no-training tests for Stage 13C selective q@k resume safety."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT / "experiments/egvan_melanoma_ablation_exp"
sys.path.insert(0, str(RUNNER_DIR))
import stage13c_selective_qk_fp32_train as runner  # noqa: E402


class FakeScaler:
    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)

    def get_scale(self):
        return self.state["scale"]


def synthetic_state():
    cfg = runner.base.variant_config("A")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    for epoch in range(1, 9):
        scheduler.step(0.1 / epoch)
    history = [{"epoch": i, "train_loss": 0.2, "val_loss": 0.1,
                "val_accuracy": 0.5, "val_balanced_accuracy": 0.5,
                "val_macro_f1": 0.1, "val_mel_precision": 0.1,
                "val_mel_recall": 0.1, "val_mel_f1": 0.1,
                "val_nv_recall": 0.1, "eligible": False,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps": 1, "amp_skips": 0} for i in range(1, 9)]
    generator = torch.Generator().manual_seed(42)
    state = {"variant": "A", "epoch": 8, "best_epoch": None,
             "best_validation_loss": math.inf, "configuration": cfg,
             "numerical_protocol": runner.PROTOCOL,
             "class_order": list(runner.base.CLASSES),
             "architecture": cfg["architecture"],
             "model_state": copy.deepcopy(model.state_dict()),
             "optimizer_state": copy.deepcopy(optimizer.state_dict()),
             "scheduler_state": copy.deepcopy(scheduler.state_dict()),
             "scaler_state": {"scale": 32768.0, "growth_factor": 2.0,
                              "backoff_factor": 0.5, "growth_interval": 2000,
                              "_growth_tracker": 8},
             "history": history,
             "sampler_generator_state": generator.get_state().clone(),
             "torch_rng_state": torch.get_rng_state().clone(),
             "cuda_rng_states": [torch.get_rng_state().clone()]}
    return state, cfg


def write_run_files(path, state, cfg):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (path / "numerical_protocol.json").write_text(json.dumps({
        "name": runner.PROTOCOL, "scientific_variant": state["variant"],
        "initial_scale": runner.INITIAL_SCALE, "gate_sha256": runner.GATE_SHA256}), encoding="utf-8")
    with (path / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state["history"][0]))
        writer.writeheader()
        writer.writerows(state["history"])


class ResumeTests(unittest.TestCase):
    def test_actual_epoch8_a_checkpoint_read_only(self):
        path = RUNNER_DIR / "stage13c_selective_qk_fp32_runs/A"
        checkpoint = path / "last_checkpoint.pt"
        if not checkpoint.is_file():
            self.skipTest("Downloaded epoch-8 A checkpoint is unavailable")
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = runner.base.variant_config("A")
        info = runner.validate_resume_checkpoint(state, "A", cfg)
        runner.validate_existing_run_files(path, state, cfg)
        self.assertEqual((info["saved_epoch"], info["start_epoch"], info["history_rows"]), (8, 9, 8))
        self.assertEqual(info["saved_amp_scale"], 32768.0)
        self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_variant_and_configuration_mismatch(self):
        state, cfg = synthetic_state()
        with self.assertRaisesRegex(ValueError, "variant"):
            runner.validate_resume_checkpoint(state, "B", runner.base.variant_config("B"))
        state["configuration"] = runner.base.variant_config("B")
        with self.assertRaisesRegex(ValueError, "configuration"):
            runner.validate_resume_checkpoint(state, "A", cfg)

    def test_nonfinite_model_optimizer_and_scaler_rejected(self):
        state, cfg = synthetic_state()
        bad = copy.deepcopy(state)
        next(iter(bad["model_state"].values())).flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "model state"):
            runner.validate_resume_checkpoint(bad, "A", cfg)
        bad = copy.deepcopy(state)
        slots = next(iter(bad["optimizer_state"]["state"].values()))
        slots["exp_inf"].flatten()[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            runner.validate_resume_checkpoint(bad, "A", cfg)
        bad = copy.deepcopy(state)
        bad["scaler_state"]["scale"] = float("nan")
        with self.assertRaisesRegex(ValueError, "GradScaler"):
            runner.validate_resume_checkpoint(bad, "A", cfg)

    def test_history_epoch_mismatch(self):
        state, cfg = synthetic_state()
        state["history"].pop()
        with self.assertRaisesRegex(ValueError, "epoch/history"):
            runner.validate_resume_checkpoint(state, "A", cfg)

    def test_missing_checkpoint_never_falls_back_to_fresh(self):
        state, cfg = synthetic_state()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            with patch.object(runner, "OUTPUT_ROOT", target):
                with self.assertRaisesRegex(FileNotFoundError, "--resume requires"):
                    runner.train_variant(ROOT, "A", cfg, resume=True)
            self.assertFalse((target / "A").exists())

    def test_scaler_rng_sampler_restoration_and_next_epoch(self):
        state, cfg = synthetic_state()
        self.assertEqual(runner.validate_resume_checkpoint(state, "A", cfg)["start_epoch"], 9)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        scaler = FakeScaler()
        generator = torch.Generator().manual_seed(999)
        captured_cuda = []
        original_rng = torch.get_rng_state().clone()
        try:
            torch.manual_seed(999)
            restored = runner.restore_resume_state(state, model, optimizer, scheduler,
                scaler, generator, cuda_rng_setter=lambda values: captured_cuda.extend(values))
            self.assertEqual(restored[0], 9)
            self.assertEqual(len(restored[1]), 8)
            self.assertEqual(scaler.get_scale(), 32768.0)
            self.assertTrue(torch.equal(generator.get_state(), state["sampler_generator_state"]))
            self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
            self.assertTrue(torch.equal(captured_cuda[0], state["cuda_rng_states"][0]))
            self.assertEqual(scheduler.state_dict(), state["scheduler_state"])
            self.assertTrue(all(torch.equal(model.state_dict()[k], v)
                                for k, v in state["model_state"].items()))
        finally:
            torch.set_rng_state(original_rng)

    def test_existing_best_is_validated_and_not_modified(self):
        state, cfg = synthetic_state()
        state["history"][2]["eligible"] = True
        state["history"][2]["val_macro_f1"] = 1.0
        state["history"][2]["val_mel_f1"] = 1.0
        state["history"][2]["val_nv_recall"] = 1.0
        state["best_epoch"] = 3
        state["best_validation_loss"] = 0.1
        best = copy.deepcopy(state)
        best["epoch"] = 3
        best["history"] = copy.deepcopy(state["history"][:3])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "A"
            write_run_files(path, state, cfg)
            torch.save(best, path / "best_checkpoint.pt")
            best_path = path / "best_checkpoint.pt"
            before = hashlib.sha256(best_path.read_bytes()).hexdigest()
            runner.validate_resume_checkpoint(state, "A", cfg)
            runner.validate_existing_run_files(path, state, cfg)
            self.assertEqual(before, hashlib.sha256(best_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
