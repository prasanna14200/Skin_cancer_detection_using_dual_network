import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.egvan import EGVAN


class EGVANTest(unittest.TestCase):
    def test_forward_and_intermediates(self):
        torch.set_num_threads(2)
        model = EGVAN(num_classes=7).eval()
        with torch.no_grad():
            for batch in (1, 2):
                x = torch.randn(batch, 3, 64, 64)
                features = model.forward_features(x)
                self.assertEqual(len(features["efficient"]), 4)
                self.assertEqual(len(features["resnet"]), 4)
                self.assertEqual(len(features["fused"]), 4)
                y = model.classifier(model.pool(features["combined"]).flatten(1))
                self.assertEqual(tuple(y.shape), (batch, 7))
                self.assertTrue(torch.isfinite(y).all())

    def test_backward_without_update(self):
        torch.set_num_threads(2)
        model = EGVAN(num_classes=7).eval()
        logits = model(torch.randn(1, 3, 64, 64))
        logits.mean().backward()
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
