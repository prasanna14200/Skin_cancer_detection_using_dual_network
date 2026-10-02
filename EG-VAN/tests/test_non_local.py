import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.egvan.non_local import NonLocalBlock


class NonLocalTest(unittest.TestCase):
    def test_shape_finite_and_gradient(self):
        for shape in ((1, 16, 3, 5), (2, 32, 6, 8), (1, 8, 1, 1)):
            x = torch.randn(shape, requires_grad=True)
            y = NonLocalBlock(shape[1])(x)
            self.assertEqual(tuple(y.shape), shape)
            self.assertTrue(torch.isfinite(y).all())
            y.mean().backward()
            self.assertIsNotNone(x.grad)
