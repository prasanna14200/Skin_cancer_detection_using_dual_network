"""CPU-only Stage25 optimizer, resume, and data-boundary checks."""
from __future__ import annotations

import ast
import copy
import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (str(ROOT / "src"), str(HERE)):
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)

import train
from comparison import compare_stage25
from initialization import MODEL_TRUNK_PREFIXES, build_two_group_optimizer, scheduler_step_preserving_ratio
from models.egvan import EGVAN


class Stage25PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        cls.model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
        cls.optimizer, cls.scheduler = build_two_group_optimizer(cls.model, cls.config)

    @classmethod
    def tearDownClass(cls):
        del cls.optimizer, cls.scheduler, cls.model

    def test_all_resnet_layers_trainable_from_epoch_one(self):
        names = dict(self.model.named_parameters())
        for prefix in MODEL_TRUNK_PREFIXES:
            matching = [p for name, p in names.items() if name.startswith(prefix)]
            self.assertTrue(matching, prefix)
            self.assertTrue(all(p.requires_grad for p in matching), prefix)
        self.assertTrue(all(p.requires_grad for p in names.values()))

    def test_no_freeze_unfreeze_or_batchnorm_eval_override(self):
        for filename in ("train.py", "initialization.py"):
            source = (HERE / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, ast.Assign) and
                                 any(isinstance(target, ast.Attribute) and
                                     target.attr == "requires_grad" for target in node.targets)
                                 for node in ast.walk(tree)), filename)
            self.assertNotIn("set_phase", source)
            self.assertNotIn("frozen_batchnorm_eval", source)
        policy = json.loads((HERE / "lr_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["frozen_layers"], [])
        self.assertFalse(policy["forced_batchnorm_eval"])
        self.model.train()
        self.assertTrue(self.model.resnet.bn1.training)
        self.assertTrue(self.model.resnet.layer1[0].bn1.training)

    def test_two_group_lr_and_complete_disjoint_coverage(self):
        groups = self.optimizer.param_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual([g["group_name"] for g in groups],
                         ["new_and_non_resnet", "pretrained_resnet_trunk"])
        self.assertEqual([g["lr"] for g in groups], [0.001, 0.0001])
        named = dict(self.model.named_parameters())
        trunk = {id(p) for name, p in named.items() if name.startswith(MODEL_TRUNK_PREFIXES)}
        other = {id(p) for name, p in named.items() if not name.startswith(MODEL_TRUNK_PREFIXES)}
        group_other = {id(p) for p in groups[0]["params"]}
        group_trunk = {id(p) for p in groups[1]["params"]}
        self.assertEqual(group_trunk, trunk)
        self.assertEqual(group_other, other)
        self.assertFalse(group_trunk & group_other)
        self.assertEqual(group_trunk | group_other, {id(p) for p in named.values()})
        self.assertEqual(sum(len(g["params"]) for g in groups), len(named))

    def test_scheduler_keeps_ten_to_one_ratio_after_reductions(self):
        tiny = torch.nn.Module()
        tiny.resnet = torch.nn.Module()
        tiny.resnet.conv1 = torch.nn.Linear(1, 1)
        tiny.resnet.bn1 = torch.nn.BatchNorm1d(1)
        for index in range(1, 5):
            setattr(tiny.resnet, f"layer{index}", torch.nn.Linear(1, 1))
        tiny.classifier = torch.nn.Linear(1, 1)
        optimizer, scheduler = build_two_group_optimizer(tiny, self.config)
        for _ in range(28):
            scheduler_step_preserving_ratio(scheduler, optimizer, 1.0)
            self.assertEqual(optimizer.param_groups[1]["lr"],
                             optimizer.param_groups[0]["lr"] * 0.1)
            self.assertEqual(scheduler.state_dict()["_last_lr"],
                             [g["lr"] for g in optimizer.param_groups])
        self.assertLess(optimizer.param_groups[0]["lr"], 0.001)

    def test_only_stage23_config_change_is_lr_policy(self):
        stage23 = json.loads((ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/config.json").read_text())
        candidate = copy.deepcopy(self.config)
        candidate["optimizer"].pop("parameter_groups")
        for item in (stage23, candidate):
            item.pop("experiment")
            item.pop("description")
        self.assertEqual(candidate, stage23)
        cfg, rule, numerical = train.read_preregistration()
        self.assertEqual(cfg, self.config)
        reference_rule = json.loads((ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/selection_rule.json").read_text())
        for key in ("melanoma_support", "stage9_correct_melanomas", "eligible_if_all",
                    "checkpoint_selection", "if_none_eligible"):
            self.assertEqual(rule[key], reference_rule[key])
        reference_numerical = json.loads((ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/numerical_protocol.json").read_text())
        for item in (numerical, reference_numerical):
            item.pop("name")
        self.assertEqual(numerical, reference_numerical)

    def test_independent_official_weight_initialization(self):
        source = (HERE / "initialization.py").read_text(encoding="utf-8")
        self.assertIn("RESNET_WEIGHTS.get_state_dict(progress=True, check_hash=True)", source)
        self.assertIn("pretrained_resnet=False", source)
        self.assertNotIn("torch.load(", source)
        self.assertEqual(train.RESNET_WEIGHTS.name, "IMAGENET1K_V2")

    def test_no_eligible_epoch_yields_no_candidate(self):
        stage15 = json.loads((ROOT / "experiments/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_metrics.json").read_text())
        stage23 = json.loads((ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json").read_text())
        self.assertEqual(compare_stage25(None, stage23, stage15)["decision"],
                         "NO_CANDIDATE_SELECTED")
        self.assertTrue(train.eligible(stage23, train.base.read_json(HERE / "selection_rule.json")))

    def test_training_loaders_exclude_ham_test_and_ph2(self):
        source = (HERE / "initialization.py").read_text(encoding="utf-8").lower()
        runner = (HERE / "train.py").read_text(encoding="utf-8").lower()
        self.assertIn('dataset(images, split, "train"', source)
        self.assertIn('dataset(images, split, "val"', source)
        for forbidden in ('dataset(images, split, "test"', 'data/external', 'ph2dataset',
                          'evaluate_ham', 'evaluate_ph2', 'stage16_final_evaluation',
                          'stage20_reliability_completion'):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, runner)

    def test_resume_restores_every_saved_rng_and_training_state(self):
        source = (HERE / "train.py").read_text(encoding="utf-8")
        for fragment in ('model.load_state_dict(state["model_state"]',
                         'optimizer.load_state_dict(state["optimizer_state"]',
                         'scheduler.load_state_dict(state["scheduler_state"]',
                         'scaler.load_state_dict(state["scaler_state"]',
                         'generator.set_state(state["sampler_generator_state"]',
                         'random.setstate(state["python_rng_state"]',
                         'np.random.set_state(state["numpy_rng_state"]',
                         'torch.set_rng_state(state["torch_rng_state"]',
                         'setter(state["cuda_rng_states"])'):
            self.assertIn(fragment, source)
        self.assertIn('return state["epoch"] + 1', source)
        self.assertIn('--resume requires existing last_checkpoint.pt; no fresh fallback', source)
        self.assertIn('"lr_new_modules": optimizer.param_groups[0]["lr"]', source)
        self.assertIn('"lr_pretrained_resnet": optimizer.param_groups[1]["lr"]', source)

    def test_resume_restores_states_and_starts_after_saved_epoch(self):
        tiny = torch.nn.Module()
        tiny.resnet = torch.nn.Module()
        tiny.resnet.conv1 = torch.nn.Linear(1, 1)
        tiny.resnet.bn1 = torch.nn.BatchNorm1d(1)
        for index in range(1, 5):
            setattr(tiny.resnet, f"layer{index}", torch.nn.Linear(1, 1))
        tiny.classifier = torch.nn.Linear(1, 1)
        optimizer, scheduler = build_two_group_optimizer(tiny, self.config)
        scaler = _CPUScaler()
        generator = torch.Generator().manual_seed(42)
        state = {
            "model_state": copy.deepcopy(tiny.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(scheduler.state_dict()),
            "scaler_state": scaler.state_dict(),
            "sampler_generator_state": generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [torch.tensor([1], dtype=torch.uint8)],
            "epoch": 4,
            "history": [{"epoch": number} for number in range(1, 5)],
            "best_epoch": None,
            "best_validation_loss": float("inf"),
        }
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        generator.manual_seed(999)
        cuda_setter = mock.Mock()
        next_epoch, history, best_epoch, best_loss = train.restore_resume(
            state, tiny, optimizer, scheduler, scaler, generator,
            cuda_rng_setter=cuda_setter)
        self.assertEqual(next_epoch, 5)
        self.assertEqual(len(history), 4)
        self.assertIsNone(best_epoch)
        self.assertEqual(best_loss, float("inf"))
        self.assertEqual(scaler.get_scale(), 8192.0)
        self.assertEqual(optimizer.param_groups[1]["lr"], optimizer.param_groups[0]["lr"] * 0.1)
        self.assertTrue(torch.equal(generator.get_state(), state["sampler_generator_state"]))
        self.assertEqual(random.getstate(), state["python_rng_state"])
        actual_np = np.random.get_state()
        saved_np = state["numpy_rng_state"]
        self.assertEqual(actual_np[0], saved_np[0])
        self.assertTrue(np.array_equal(actual_np[1], saved_np[1]))
        self.assertEqual(actual_np[2:], saved_np[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
        cuda_setter.assert_called_once()


class _CPUScaler:
    def __init__(self):
        self.value = 8192.0

    def state_dict(self):
        return {"scale": self.value}

    def load_state_dict(self, state):
        self.value = float(state["scale"])

    def get_scale(self):
        return self.value


if __name__ == "__main__":
    unittest.main()
