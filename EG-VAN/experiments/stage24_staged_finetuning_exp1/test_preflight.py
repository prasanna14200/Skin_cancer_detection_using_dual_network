"""CPU-only Stage24 policy and resume tests; no data loaders or pretrained downloads."""
from __future__ import annotations

import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch

import train as runner
from finetuning import Stage24EGVAN, apply_scheduler_step, assert_optimizer_layout, build_optimizer_scheduler


class Stage24PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = runner.base.read_json(runner.HERE / "config.json")

    def setUp(self):
        self.model = Stage24EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
        self.optimizer, self.scheduler = build_optimizer_scheduler(self.model, self.config)
        assert_optimizer_layout(self.model, self.optimizer)

    def tearDown(self):
        del self.model, self.optimizer, self.scheduler

    def test_phase_boundaries_and_batchnorm(self):
        for saved, next_epoch, next_phase, layer2 in (
            (4, 5, "A", False), (5, 6, "B", True), (6, 7, "B", True)
        ):
            with self.subTest(saved_epoch=saved):
                self.assertEqual(Stage24EGVAN.phase_for_next_epoch(saved), next_phase)
                state = self.model.set_phase(next_epoch)
                self.model.train(True)
                self.assertEqual(state["fine_tuning_phase"], next_phase)
                self.assertEqual(state["layer2_trainable"], layer2)
                self.assertFalse(state["conv1_trainable"])
                self.assertFalse(state["bn1_trainable"])
                self.assertFalse(state["layer1_trainable"])
                self.assertTrue(state["layer3_trainable"])
                self.assertTrue(state["layer4_trainable"])
                self.assertTrue(all(self.model.frozen_batchnorm_state().values()))
                self.assertFalse(self.model.resnet.bn1.training)
                self.assertTrue(all(not module.training for module in self.model.resnet.layer1.modules()
                                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)))
                layer2_bn = [module for module in self.model.resnet.layer2.modules()
                             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
                self.assertTrue(layer2_bn)
                self.assertTrue(all(module.training == layer2 for module in layer2_bn))

    def test_frozen_batchnorm_running_statistics(self):
        self.model.set_phase(1)
        self.model.train(True)
        frozen = [self.model.resnet.bn1]
        frozen += [m for m in self.model.resnet.layer1.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        frozen += [m for m in self.model.resnet.layer2.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        before = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in frozen]
        with torch.no_grad():
            for module in frozen:
                module(torch.randn(2, module.num_features, 2, 2))
        for module, expected in zip(frozen, before):
            for actual, saved in zip((module.running_mean, module.running_var, module.num_batches_tracked), expected):
                self.assertTrue(torch.equal(actual, saved))

    def test_optimizer_unfreeze_preserves_state_and_lr_ratios(self):
        layer3 = next(self.model.resnet.layer3.parameters())
        layer2 = next(self.model.resnet.layer2.parameters())
        non_resnet = self.model.classifier.weight
        self.model.set_phase(1)
        self.optimizer.zero_grad(set_to_none=True)
        layer3.grad = torch.full_like(layer3, 0.01)
        non_resnet.grad = torch.full_like(non_resnet, 0.01)
        self.optimizer.step()
        self.assertNotIn(layer2, self.optimizer.state)
        old_layer3_state = {k: v.clone() if torch.is_tensor(v) else v
                            for k, v in self.optimizer.state[layer3].items()}
        old_nonresnet_state = {k: v.clone() if torch.is_tensor(v) else v
                               for k, v in self.optimizer.state[non_resnet].items()}
        optimizer_id = id(self.optimizer)
        scheduler_id = id(self.scheduler)
        self.model.set_phase(6)
        self.assertEqual(id(self.optimizer), optimizer_id)
        self.assertEqual(id(self.scheduler), scheduler_id)
        self.assertTrue(layer2.requires_grad)
        for param, old in ((layer3, old_layer3_state), (non_resnet, old_nonresnet_state)):
            for key, value in old.items():
                actual = self.optimizer.state[param][key]
                self.assertTrue(torch.equal(actual, value) if torch.is_tensor(value) else actual == value)
        self.optimizer.zero_grad(set_to_none=True)
        layer2.grad = torch.full_like(layer2, 0.01)
        self.optimizer.step()
        self.assertIn(layer2, self.optimizer.state)
        for _ in range(3):
            rates = apply_scheduler_step(self.scheduler, self.optimizer, 1.0)
            self.assertEqual(rates, [rates[0], rates[0] * .1, rates[0] * .1, rates[0] * .05])
        self.assertLess(rates[0], 0.001)
        saved_scheduler = copy.deepcopy(self.scheduler.state_dict())
        self.scheduler.step(0.5)
        self.scheduler.load_state_dict(saved_scheduler)
        self.assertEqual(self.scheduler.state_dict(), saved_scheduler)

    def test_resume_restores_phase_scaler_and_rng(self):
        scaler = torch.amp.GradScaler("cpu", init_scale=16384)
        generator = torch.Generator(device="cpu").manual_seed(42)
        for saved_epoch in (4, 5, 6):
            with self.subTest(saved_epoch=saved_epoch):
                self.model.set_phase(saved_epoch)
                state = {
                    "model_state": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
                    "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
                    "scaler_state": copy.deepcopy(scaler.state_dict()),
                    "sampler_generator_state": generator.get_state().clone(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state().clone(),
                    "cuda_rng_states": [torch.tensor([1, 2], dtype=torch.uint8)],
                    "optimizer_group_lrs": [g["lr"] for g in self.optimizer.param_groups],
                }
                next_epoch = saved_epoch + 1
                with mock.patch.object(runner, "validate_checkpoint_state", return_value={
                    "next_epoch": next_epoch, "next_phase": Stage24EGVAN.phase_for_epoch(next_epoch)
                }), mock.patch.object(runner.torch.cuda, "set_rng_state_all") as cuda_restore:
                    result = runner.prepare_resume_state(
                        state, self.model, self.optimizer, self.scheduler, scaler,
                        generator, self.config, {})
                self.assertEqual(result["next_epoch"], next_epoch)
                self.assertEqual(self.model.trainability_state()["layer2_trainable"], next_epoch >= 6)
                self.assertEqual(scaler.get_scale(), 16384)
                self.assertTrue(torch.equal(generator.get_state(), state["sampler_generator_state"]))
                self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_rng_state"]))
                cuda_restore.assert_called_once()

    def test_checkpoint_validation_and_rejections(self):
        self.model.set_phase(5)
        self.model.train(True)
        for _ in range(5):
            apply_scheduler_step(self.scheduler, self.optimizer, 0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8192)
        rows = [{
            "epoch": epoch, "train_loss": 0.1, "val_loss": 0.1,
            "val_accuracy": 0.8, "val_macro_f1": 0.6,
            "val_mel_tp": 0, "val_mel_recall": 0.0, "val_mel_f1": 0.0,
            "val_nv_recall": 0.9, "eligible": False,
            "fine_tuning_phase": "A",
        } for epoch in range(1, 6)]
        state = {
            "experiment": runner.NAME, "epoch": 5,
            "configuration": self.config,
            "config_sha256": runner.CONFIG_SHA,
            "selection_rule_sha256": runner.RULE_SHA,
            "numerical_protocol_sha256": runner.NUMERICAL_SHA,
            "finetuning_policy_sha256": runner.FINETUNING_POLICY_SHA,
            "initialization_spec_sha256": runner.INITIALIZATION_SPEC_SHA,
            "reference_hashes": runner.FROZEN_REFERENCE_HASHES,
            "runner_sha256": "stub",
            "class_order": self.config["classes"],
            "architecture": self.config["architecture"],
            "architecture_metadata": {"test_only": True},
            "numerical_protocol": runner.base.read_json(runner.HERE / "numerical_protocol.json"),
            "finetuning_policy": runner.base.read_json(runner.HERE / "finetuning_policy.json"),
            "initialization_spec": runner.base.read_json(runner.HERE / "initialization_spec.json"),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": rows, "best_epoch": None, "best_validation_loss": float("inf"),
            "fine_tuning_phase": "A", "trainability_state": self.model.trainability_state(),
            "frozen_batchnorm_state": self.model.frozen_batchnorm_state(),
            "optimizer_group_lrs": [g["lr"] for g in self.optimizer.param_groups],
            "optimizer_group_multipliers": [g["lr_multiplier"] for g in self.optimizer.param_groups],
            "initialization_report": {"weight_enum": "ResNet50_Weights.IMAGENET1K_V2"},
            "initialization_report_file_sha256": "stub",
            "numerical_events": [],
            "validation_artifacts": {e: {} for e in range(1, 6)},
            "latest_validation_payload": {"predictions": [], "metrics": {}},
            "sampler_generator_state": torch.Generator().manual_seed(42).get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [torch.tensor([1, 2], dtype=torch.uint8)],
        }
        rule = runner.base.read_json(runner.HERE / "selection_rule.json")
        with mock.patch.object(runner, "sha256", return_value="stub"):
            report = runner.validate_checkpoint_state(state, self.config, rule)
            self.assertEqual((report["saved_epoch"], report["next_epoch"], report["next_phase"]),
                             (5, 6, "B"))
            bad = dict(state, epoch=4)
            with self.assertRaisesRegex(ValueError, "epoch/history"):
                runner.validate_checkpoint_state(bad, self.config, rule)
            bad = dict(state, experiment="wrong")
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.validate_checkpoint_state(bad, self.config, rule)
            bad = dict(state, scaler_state={})
            with self.assertRaisesRegex(ValueError, "GradScaler"):
                runner.validate_checkpoint_state(bad, self.config, rule)
            corrupt_model = dict(state["model_state"])
            corrupt_model["classifier.weight"] = torch.full_like(corrupt_model["classifier.weight"], float("nan"))
            bad = dict(state, model_state=corrupt_model)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                runner.validate_checkpoint_state(bad, self.config, rule)


if __name__ == "__main__":
    unittest.main()
