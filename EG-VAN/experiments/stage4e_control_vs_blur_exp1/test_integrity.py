"""CPU-safe tests: Stage 4E provenance, paired factors, selection and resume."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torchvision import transforms
from pathlib import Path

import common
import evaluate_pair
train = evaluate_pair.runner


class _FakeEGVAN(torch.nn.Module):
    def __init__(self, num_classes, pretrained_efficient, pretrained_resnet):
        super().__init__()
        self.resnet = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, num_classes)


class _FakeWeights:
    name = "IMAGENET1K_V2"
    url = "test://official-weight-fixture"

    def get_state_dict(self, progress, check_hash):
        if not check_hash:
            raise AssertionError("Official weight hash check disabled")
        return {}


class _FakeScaler:
    def __init__(self):
        self.value = 8192.0

    def load_state_dict(self, state):
        self.value = state["scale"]

    def get_scale(self):
        return self.value


class Stage4EIntegrityTests(unittest.TestCase):
    def test_exact_v2_boundary_and_no_original_evaluation_membership(self):
        result = common.preflight(common.ROOT, hash_images=False)
        self.assertEqual(result["status"], "PREFLIGHT_PASS")
        self.assertEqual((result["train_images"], result["selection_images"]), (6477, 1538))
        self.assertEqual(result["selection_mel_images"], 163)

    def test_arm_config_differs_only_by_registered_blur_flag(self):
        cfg = common.read_config()
        a = common.arm_config(cfg, "control")
        b = common.arm_config(cfg, "treatment")
        self.assertEqual({k: v for k, v in a.items() if k not in ("arm", "blur_enabled")},
                         {k: v for k, v in b.items() if k not in ("arm", "blur_enabled")})
        self.assertFalse(a["blur_enabled"])
        self.assertTrue(b["blur_enabled"])
        self.assertEqual(cfg["treatment_only_transform"],
                         {"position": "after_RandomRotation_before_ToTensor", "type": "RandomApply_GaussianBlur",
                          "probability": 0.25, "kernel_size": 3, "sigma_min": 0.1, "sigma_max": 0.5})

    def test_transform_isolation_and_identical_selection_path(self):
        control_train, control_selection = common.transforms_for_arm("control")
        treatment_train, treatment_selection = common.transforms_for_arm("treatment")
        self.assertEqual(repr(control_selection), repr(treatment_selection))
        self.assertNotIn("GaussianBlur", repr(control_train))
        self.assertNotIn("GaussianBlur", repr(control_selection))
        self.assertNotIn("GaussianBlur", repr(treatment_selection))
        layers = treatment_train.transforms
        self.assertIsInstance(layers[3], transforms.RandomRotation)
        self.assertIsInstance(layers[4], transforms.RandomApply)
        self.assertEqual(layers[4].p, 0.25)
        self.assertEqual(layers[4].transforms[0].kernel_size, (3, 3))
        self.assertEqual(layers[4].transforms[0].sigma, (0.1, 0.5))
        self.assertIsInstance(layers[5], transforms.ToTensor)
        self.assertEqual(repr(control_train.transforms[:4]), repr(layers[:4]))
        self.assertEqual(repr(control_train.transforms[-2:]), repr(layers[-2:]))

    def test_two_fresh_initializations_match_without_trained_checkpoint(self):
        with (mock.patch.object(common, "EGVAN", _FakeEGVAN),
              mock.patch.object(common, "RESNET_WEIGHTS", _FakeWeights()),
              mock.patch.object(common, "apply_resnet_state", return_value={"status": "LOADED"})):
            a, first = common.fresh_model()
            b, second = common.fresh_model()
        self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
        self.assertEqual(common.model_fingerprint(a), common.model_fingerprint(b))
        with torch.no_grad():
            b.head.weight[0, 0] += 1
        self.assertNotEqual(common.model_fingerprint(a), common.model_fingerprint(b))

    def test_sampler_and_training_pool_match_registered_policy(self):
        rows = common.read_csv(common.V2 / common.TRAIN_NAME)
        generator = torch.Generator().manual_seed(42)
        sampler = common.sampler_for(rows, generator)
        self.assertEqual(sampler.num_samples, 6477)
        self.assertTrue(sampler.replacement)
        self.assertEqual(float(sampler.weights[[i for i, row in enumerate(rows) if row["dx"] == "mel"][0]]), 1.5630495442733532)

    def test_candidate_gate_is_not_accuracy_only(self):
        deltas = {"mel_recall": 0.05, "macro_f1": 0.0, "mel_specificity": 0.0}
        intervals = {"mel_recall": {"lower_95": 0.001}}
        self.assertTrue(evaluate_pair.candidate_gate(deltas, intervals))
        self.assertFalse(evaluate_pair.candidate_gate({**deltas, "macro_f1": -0.001}, intervals))
        self.assertFalse(evaluate_pair.candidate_gate({**deltas, "mel_specificity": -0.001}, intervals))
        self.assertFalse(evaluate_pair.candidate_gate(deltas, {"mel_recall": {"lower_95": 0.0}}))

    def test_lesion_paired_bootstrap_uses_both_arms_same_ids(self):
        source, control, treatment = [], [], []
        for name in common.CLASSES:
            for index in range(2):
                image_id = f"{name}_{index}"
                source.append({"image_id": image_id, "lesion_id": image_id, "dx": name})
                control.append({"image_id": image_id, "true_label": name,
                                "predicted_label": "nv" if name == "mel" and index == 0 else name})
                treatment.append({"image_id": image_id, "true_label": name, "predicted_label": name})
        a = evaluate_pair.paired_bootstrap(control, treatment, source, repetitions=30)
        b = evaluate_pair.paired_bootstrap(control, treatment, source, repetitions=30)
        self.assertEqual(a, b)
        self.assertIn("mel_recall", a)
        treatment[0]["image_id"] = "changed"
        with self.assertRaises(ValueError):
            evaluate_pair.paired_bootstrap(control, treatment, source, repetitions=2)

    def test_resume_validation_and_rng_restore(self):
        cfg = common.read_config()
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        scheduler.step(0.1)
        sampler = torch.Generator().manual_seed(42)
        scaler = _FakeScaler()
        state = {
            "arm": "control", "epoch": 1, "configuration": common.arm_config(cfg, "control"),
            "config_sha256": common.CONFIG_SHA, "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
            "runner_sha256": common.sha256(Path(train.__file__)),
            "common_sha256": common.sha256(common.HERE / "common.py"),
            "initial_state_sha256": "fixture_initial_hash", "model_state": copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(scheduler.state_dict()),
            "scaler_state": {"scale": 8192.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                             "growth_interval": 2000, "_growth_tracker": 0},
            "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}],
            "best_epoch": 1, "best_validation_loss": 0.1,
            "sampler_generator_state": sampler.get_state(),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": [], "numerical_events": [],
        }
        receipt = {"initial_state_sha256": "fixture_initial_hash"}
        train.check_resume_state(state, cfg, "control", receipt, 1)
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        sampler.manual_seed(99)
        setter = mock.Mock()
        next_epoch, history, events = train.restore_training_state(
            state, model, optimizer, scheduler, scaler, sampler, cuda_rng_setter=setter)
        self.assertEqual((next_epoch, len(history), len(events)), (2, 1, 0))
        self.assertEqual(random.getstate(), state["python_rng_state"])
        self.assertTrue(np.array_equal(np.random.get_state()[1], state["numpy_rng_state"][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
        self.assertTrue(torch.equal(sampler.get_state(), state["sampler_generator_state"]))
        setter.assert_called_once()
        bad = copy.deepcopy(state)
        bad["model_state"]["weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            train.check_resume_state(bad, cfg, "control", receipt, 1)
        bad = copy.deepcopy(state)
        bad["epoch"] = 2
        with self.assertRaises(ValueError):
            train.check_resume_state(bad, cfg, "control", receipt, 2)

    def test_no_old_evaluation_loader_or_stage23_trained_checkpoint(self):
        source = (common.HERE / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("best_checkpoint.pt", source)
        self.assertNotIn("data/external", source)
        self.assertNotIn("PH2Dataset", source)
        self.assertIn("common.V2 / common.TRAIN_NAME", source)
        self.assertIn("common.V2 / common.SELECT_NAME", source)
        self.assertNotIn("load_frozen", source)
        self.assertNotIn("validation_predictions.csv", source)
        self.assertEqual(Path(train.__file__).resolve(), (common.HERE / "train.py").resolve())

    def test_epoch_commit_is_atomic_immutable_and_hash_verified(self):
        cfg = common.read_config()
        expected = [
            {"image_id": name, "lesion_id": f"lesion-{name}", "dx": name}
            for name in common.CLASSES
        ]
        predictions = [
            {"image_id": name, "true_label": name, "predicted_label": name,
             "probabilities": json.dumps([1.0 if item == name else 0.0 for item in common.CLASSES])}
            for name in common.CLASSES
        ]
        metrics = common.metrics_from_prediction_rows(predictions)
        evidence = {
            **metrics, "epoch": 1, "validation_loss": 0.1, "arm": "control",
            "class_order": list(common.CLASSES), "mel_tp": 1, "mel_fn": 0,
        }
        state = {
            "arm": "control", "epoch": 1, "configuration": common.arm_config(cfg, "control"),
            "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1}],
            "numerical_events": [],
        }
        original_read_csv = common.read_csv

        def read_csv(path):
            if Path(path) == common.V2 / common.SELECT_NAME:
                return expected
            return original_read_csv(path)

        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "control"
            with mock.patch.object(common, "read_csv", side_effect=read_csv):
                commit = train.save_epoch(arm_dir, state, predictions, evidence)
                folder = train.epoch_folder(arm_dir, 1)
                self.assertEqual(
                    common.sha256(folder / "checkpoint.pt"),
                    commit["artifact_sha256"]["checkpoint.pt"],
                )
                with self.assertRaises(FileExistsError):
                    train.save_epoch(arm_dir, state, predictions, evidence)
                (folder / "checkpoint.pt").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    train.verify_epoch(folder, state["configuration"], "control")

    def test_interrupted_epoch_is_not_discoverable_as_completed(self):
        state = {"epoch": 1}
        with tempfile.TemporaryDirectory() as directory:
            arm_dir = Path(directory) / "control"
            with mock.patch.object(train.torch, "save", side_effect=OSError("simulated interruption")):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    train.save_epoch(arm_dir, state, [{"image_id": "fixture"}], {"epoch": 1})
            self.assertFalse(train.epoch_folder(arm_dir, 1).exists())
            self.assertFalse(list((arm_dir / "epochs").glob("epoch_[0-9][0-9][0-9]")))
            with self.assertRaisesRegex(FileNotFoundError, "completed immutable epoch"):
                train.load_resume(arm_dir, common.read_config(), "control", {"initial_state_sha256": "x"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
