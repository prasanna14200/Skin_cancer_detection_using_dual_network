"""Local, synthetic Stage 13 checks; never reads HAM test or PH2."""
import inspect
import math
import unittest

import torch

import train_ablation as stage13


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_intended_config_leaves_change(self):
        a, b, c = (stage13.variant_config(x) for x in "ABC")
        def comparable(cfg):
            return {k: v for k, v in cfg.items() if k not in ("variant", "description")}
        a, b, c = map(comparable, (a, b, c))
        b_loss = b.pop("loss")
        c_sampler = c.pop("sampler")
        self.assertEqual(a | {"loss": b_loss}, b | {"loss": b_loss})
        self.assertEqual(a | {"sampler": c_sampler}, c | {"sampler": c_sampler})
        self.assertEqual(b_loss["mel_multiplier"], (5366 / 899) ** 0.25)
        self.assertEqual(c_sampler["mel_weight"], (5366 / 899) ** 0.5)

    def test_control_focal_exact_and_mel_only_multiplier(self):
        _, _, _, _, baseline_loss, _, _ = stage13.imports(stage13.ROOT)
        logits = torch.tensor([[0.2, -0.1, 0.3, 0.5, 1.0, 0.4, -0.2],
                               [1.2, 0.7, -0.1, 0.0, -0.3, 0.2, -0.5]], requires_grad=True)
        targets = torch.tensor([4, 0])
        control = stage13.loss(logits, targets)
        torch.testing.assert_close(control, baseline_loss(logits, targets))
        multiplier = stage13.variant_config("B")["loss"]["mel_multiplier"]
        mel_single = baseline_loss(logits[:1], targets[:1])
        other_single = baseline_loss(logits[1:], targets[1:])
        torch.testing.assert_close(stage13.loss(logits, targets, multiplier),
                                   (multiplier * mel_single + other_single) / 2)
        stage13.loss(logits, targets, multiplier).backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_frozen_selection_gate_and_tiebreak(self):
        ref = {"melanoma_recall": 62/107, "accuracy": 0.8225152129817445,
               "macro_f1": 0.6709124104985493, "nevus_recall": 0.947209653092006}
        base = {"accuracy": .82, "macro_f1": .67, "nv_recall": .94,
                "mel_f1": .60, "validation_loss": .08}
        a = {"variant": "A", "mel_recall": 62/107, **base}
        b = {"variant": "B", "mel_recall": 63/107, **base}
        c = {"variant": "C", "mel_recall": 63/107, **base}
        selected, eligible = stage13.select_candidate([a,b,c], ref)
        self.assertEqual([x["variant"] for x in eligible], ["B","C"])
        self.assertEqual(selected["variant"], "B")
        c["mel_recall"] = 64/107
        self.assertEqual(stage13.select_candidate([a,b,c], ref)[0]["variant"], "C")
        c["macro_f1"] = .60
        b["mel_recall"] = 62/107
        self.assertIsNone(stage13.select_candidate([a,b,c], ref)[0])

    def test_training_path_excludes_locked_loaders(self):
        source = inspect.getsource(stage13.train)
        self.assertIn('Dataset(images, split, "train"', source)
        self.assertIn('Dataset(images, split, "val"', source)
        self.assertNotIn('Dataset(images, split, "test"', source)
        self.assertNotIn('external/ph2', source.lower())

    def test_synthetic_model_optimizer_scheduler(self):
        _, _, _, EGVAN, _, _, _ = stage13.imports(stage13.ROOT)
        model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 58087409)
        model.train()
        x = torch.randn(2,3,64,64)
        y = torch.tensor([4,0])
        optimizer, scheduler = stage13.optimizer_scheduler(model)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2,7))
        value = stage13.loss(logits,y,stage13.variant_config("B")["loss"]["mel_multiplier"])
        self.assertTrue(math.isfinite(value.item()))
        value.backward(); optimizer.step(); scheduler.step(value.item())
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
