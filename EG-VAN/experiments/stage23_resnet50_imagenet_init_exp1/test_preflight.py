"""CPU/read-only tests for the Stage23 initialization and safety contract."""
from __future__ import annotations

import ast
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torchvision.models import ResNet50_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
for entry in (str(SRC), str(HERE)):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import train
from comparison import compare_validation
from models.egvan import EGVAN
from models.egvan.modified_resnet50 import ModifiedResNet50


class Stage23PreflightTests(unittest.TestCase):
    def test_stage15_builder_uses_random_resnet(self):
        source = (ROOT / "experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py").read_text(encoding="utf-8")
        self.assertIn("EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False)", source)
        self.assertIn("resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)",
                      (SRC / "models/egvan/modified_resnet50.py").read_text(encoding="utf-8"))

    def test_stage23_uses_explicit_imagenet_v2_enum(self):
        spec = json.loads((HERE / "initialization_spec.json").read_text(encoding="utf-8"))
        self.assertIs(ResNet50_Weights.IMAGENET1K_V2, train.RESNET_WEIGHTS)
        self.assertEqual(spec["official_weight_url"], train.RESNET_WEIGHTS.url)
        self.assertEqual(spec["loader"], train.RESNET_LOADER_NAME)
        self.assertIn("get_state_dict(progress=True, check_hash=True)",
                      (HERE / "initialization.py").read_text(encoding="utf-8"))

    def test_stage23_config_changes_only_resnet_initialization(self):
        cfg, _, _ = train.read_preregistration()
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/config.json")
        candidate = copy.deepcopy(cfg)
        candidate.pop("experiment")
        candidate.pop("description")
        candidate.pop("data")
        candidate["initialization"]["resnet50"] = "random"
        reference.pop("experiment")
        reference.pop("description")
        self.assertEqual(candidate, reference)

    def test_only_resnet_backbone_state_changes(self):
        torch.manual_seed(42)
        source_model = ModifiedResNet50(pretrained=False).eval()
        torch.manual_seed(42)
        target_model = ModifiedResNet50(pretrained=False).eval()
        self.assertEqual(source_model.state_dict().keys(), target_model.state_dict().keys())
        source_state = {key: torch.full_like(value, 0.125) if value.is_floating_point()
                        else torch.zeros_like(value) for key, value in source_model.state_dict().items()
                        if key.startswith(train.BACKBONE_PREFIXES)}
        source_state["fc.weight"] = torch.zeros(1000, 2048)
        source_state["fc.bias"] = torch.zeros(1000)
        report = train.apply_resnet_state(target_model, source_state)
        self.assertEqual(report["shape_mismatches"], [])
        self.assertEqual(report["unexpected_source_keys"], [])
        expected_new = [key for key in target_model.state_dict()
                        if not key.startswith(train.BACKBONE_PREFIXES)]
        self.assertEqual(report["new_module_keys"], sorted(expected_new))
        changed = [key for key in source_model.state_dict()
                   if not torch.equal(source_model.state_dict()[key], target_model.state_dict()[key])]
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith(train.BACKBONE_PREFIXES) for key in changed))

    def test_resnet_output_shapes_unchanged(self):
        torch.manual_seed(42)
        model = ModifiedResNet50(pretrained=False).eval()
        source_state = {key: torch.full_like(value, 0.125) if value.is_floating_point()
                        else torch.zeros_like(value) for key, value in model.state_dict().items()
                        if key.startswith(train.BACKBONE_PREFIXES)}
        source_state["fc.weight"] = torch.zeros(1000, 2048)
        source_state["fc.bias"] = torch.zeros(1000)
        train.apply_resnet_state(model, source_state)
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual([tuple(x.shape) for x in outputs], [
            (1, 256, 16, 16), (1, 512, 8, 8), (1, 1024, 4, 4), (1, 2048, 2, 2)
        ])

    def test_full_egvan_feature_shapes_and_class_count_unchanged(self):
        model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False).eval()
        with torch.inference_mode():
            features = model.forward_features(torch.zeros(1, 3, 64, 64))
            logits = model.classifier(model.pool(features["combined"]).flatten(1))
        self.assertEqual([tuple(x.shape) for x in features["efficient"]], [
            (1, 48, 16, 16), (1, 64, 8, 8), (1, 160, 4, 4), (1, 1280, 2, 2)
        ])
        self.assertEqual([tuple(x.shape) for x in features["resnet"]], [
            (1, 256, 16, 16), (1, 512, 8, 8), (1, 1024, 4, 4), (1, 2048, 2, 2)
        ])
        self.assertEqual([tuple(x.shape) for x in features["fused"]], [
            (1, 128, 8, 8), (1, 128, 4, 4), (1, 128, 2, 2), (1, 128, 1, 1)
        ])
        self.assertEqual(tuple(logits.shape), (1, 7))

    def test_class_order_unchanged(self):
        cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(cfg["classes"]), tuple(train.base.CLASSES))
        self.assertEqual(tuple(cfg["classes"]), ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))

    def _stage15_validation_metrics(self):
        reference_dir = ROOT / "experiments/stage15_single_candidate_exp1"
        path = reference_dir / "run/validation_epochs/epoch_016_metrics.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_comparison_decision_no_candidate(self):
        self.assertEqual(compare_validation(None, self._stage15_validation_metrics())["decision"],
                         "NO_CANDIDATE_SELECTED")

    def test_comparison_decision_no_meaningful_improvement(self):
        reference = self._stage15_validation_metrics()
        candidate = copy.deepcopy(reference)
        candidate["experiment"] = "stage23_resnet50_imagenet_init_exp1"
        self.assertEqual(compare_validation(candidate, reference)["decision"],
                         "NO_MEANINGFUL_IMPROVEMENT")

    def test_comparison_decision_clear_improvement(self):
        reference = self._stage15_validation_metrics()
        candidate = copy.deepcopy(reference)
        candidate["experiment"] = "stage23_resnet50_imagenet_init_exp1"
        candidate["accuracy"] += 0.001
        result = compare_validation(candidate, reference)
        self.assertEqual(result["decision"], "CLEAR_VALIDATION_IMPROVEMENT")

    def test_comparison_decision_mixed_result(self):
        reference = self._stage15_validation_metrics()
        candidate = copy.deepcopy(reference)
        candidate["experiment"] = "stage23_resnet50_imagenet_init_exp1"
        candidate["accuracy"] += 0.001
        candidate["per_class"]["mel"]["precision"] -= 0.01
        result = compare_validation(candidate, reference)
        self.assertEqual(result["decision"], "MIXED_VALIDATION_RESULT")

    def test_split_hash_is_frozen(self):
        cfg = train.base.read_json(HERE / "config.json")
        split = ROOT / "data/splits/split_leakage_aware.csv"
        self.assertEqual(train.base.sha256(split), train.SPLIT_SHA)
        self.assertEqual(train.SPLIT_SHA, cfg["data"]["split_sha256"])

    def test_preprocessing_source_hash_is_frozen(self):
        self.assertEqual(train.base.sha256(SRC / "preprocessing.py"), train.FROZEN_SOURCE_HASHES["src/preprocessing.py"])
        self.assertEqual(train.base.sha256(SRC / "dataset.py"), train.FROZEN_SOURCE_HASHES["src/dataset.py"])

    def test_sampler_matches_stage15(self):
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/config.json")
        candidate = train.base.read_json(HERE / "config.json")
        self.assertEqual(candidate["sampler"], reference["sampler"])
        self.assertEqual(candidate["sampler"]["mel_weight"], 1.5630495442733532)

    def test_focal_loss_settings_match_stage15(self):
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/config.json")
        candidate = train.base.read_json(HERE / "config.json")
        self.assertEqual({key: value for key, value in candidate["loss"].items() if key != "mel_multiplier"},
                         {key: value for key, value in reference["loss"].items() if key != "mel_multiplier"})

    def test_mel_multiplier_matches_stage15(self):
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/config.json")
        candidate = train.base.read_json(HERE / "config.json")
        self.assertEqual(candidate["loss"]["mel_multiplier"], 1.2815247721366766)
        self.assertEqual(candidate["loss"]["mel_multiplier"], reference["loss"]["mel_multiplier"])

    def test_numerical_policy_matches_stage15_except_name(self):
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/numerical_protocol.json")
        candidate = train.base.read_json(HERE / "numerical_protocol.json")
        reference.pop("name")
        candidate.pop("name")
        self.assertEqual(candidate, reference)

    def test_selection_gates_match_stage15_exactly(self):
        reference = train.base.read_json(ROOT / "experiments/stage15_single_candidate_exp1/selection_rule.json")
        candidate = train.base.read_json(HERE / "selection_rule.json")
        for key in ("melanoma_support", "stage9_correct_melanomas", "eligible_if_all", "checkpoint_selection", "if_none_eligible"):
            self.assertEqual(candidate[key], reference[key])

    def test_stage15_nonlocal_precision_policy_is_installed(self):
        source = (HERE / "train.py").read_text(encoding="utf-8")
        self.assertIn("install_selective_qk_policy(model)", source)
        self.assertEqual(train.base.sha256(ROOT / "experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_train.py"),
                         train.FROZEN_SOURCE_HASHES[
                             "experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_train.py"])

    def test_ham_test_is_inaccessible_to_training_loader(self):
        source = (HERE / "initialization.py").read_text(encoding="utf-8").lower()
        self.assertNotIn('dataset(images, split, "test"', source)
        self.assertNotIn("evaluate_ham", source)

    def test_ph2_is_inaccessible_to_training_loader(self):
        source = (HERE / "initialization.py").read_text(encoding="utf-8").lower()
        for forbidden in ("evaluate_ph2", "data/external", "ph2dataset"):
            self.assertNotIn(forbidden, source)

    def test_restore_resume_restores_rng_and_training_states(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        scaler = _CPUScaler()
        generator = torch.Generator().manual_seed(11)
        state = {
            "model_state": copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(scheduler.state_dict()),
            "scaler_state": scaler.state_dict(),
            "sampler_generator_state": generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [torch.tensor([1], dtype=torch.uint8)],
            "epoch": 3,
            "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            "best_epoch": 2,
            "best_validation_loss": 0.1,
        }
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        generator.manual_seed(999)
        restore_cuda = mock.Mock()
        next_epoch, history, best_epoch, best_loss = train.restore_resume(
            state, model, optimizer, scheduler, scaler, generator, cuda_rng_setter=restore_cuda
        )
        self.assertEqual((next_epoch, best_epoch, best_loss), (4, 2, 0.1))
        self.assertEqual(len(history), 3)
        self.assertTrue(torch.equal(generator.get_state(), state["sampler_generator_state"]))
        self.assertEqual(random.getstate(), state["python_rng_state"])
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], state["numpy_rng_state"][0])
        self.assertTrue(np.array_equal(actual_numpy[1], state["numpy_rng_state"][1]))
        self.assertEqual(actual_numpy[2:], state["numpy_rng_state"][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
        restore_cuda.assert_called_once()
        self.assertEqual(scaler.get_scale(), 8192.0)

    def test_no_candidate_finalization_does_not_create_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            best_path = Path(temporary) / "best_checkpoint.pt"
            self.assertIsNone(train.best_checkpoint_for_finalization(None, best_path))
            self.assertFalse(best_path.exists())


class _CPUScaler:
    def __init__(self):
        self.scale_value = 8192.0

    def state_dict(self):
        return {"scale": self.scale_value, "growth_factor": 2.0, "backoff_factor": 0.5,
                "growth_interval": 2000, "_growth_tracker": 0}

    def load_state_dict(self, state):
        self.scale_value = float(state["scale"])

    def get_scale(self):
        return self.scale_value


if __name__ == "__main__":
    unittest.main(verbosity=2)
