import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.egvan.scga import GroupMeanMaxAttention, SCGA


class SCGATest(unittest.TestCase):
    def test_shapes_and_gradients(self):
        for shape in ((1, 64, 9, 11), (2, 128, 6, 8)):
            x = torch.randn(shape, requires_grad=True)
            y = SCGA(shape[1])(x)
            self.assertEqual(tuple(y.shape), shape)
            self.assertTrue(torch.isfinite(y).all())
            y.mean().backward()
            self.assertIsNotNone(x.grad)

    def test_group_requirement(self):
        with self.assertRaises(ValueError):
            GroupMeanMaxAttention(65, groups=8)
