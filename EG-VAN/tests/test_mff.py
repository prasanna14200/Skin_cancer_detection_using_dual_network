import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.egvan.mff import MFF


class MFFTest(unittest.TestCase):
    def test_actual_backbone_pair_widths(self):
        module = MFF(48, 256, out_channels=64).eval()
        with torch.no_grad():
            output = module(torch.randn(1, 48, 24, 24), torch.randn(1, 256, 24, 24))
        self.assertEqual(tuple(output.shape), (1, 64, 12, 12))

    def test_explicit_alignment(self):
        module = MFF(16, 32, out_channels=32).eval()
        with torch.no_grad():
            output = module(torch.randn(1, 16, 15, 17), torch.randn(1, 32, 16, 18))
        self.assertEqual(tuple(output.shape), (1, 32, 8, 9))

    def test_serial_carry_and_backward(self):
        module = MFF(16, 32, out_channels=32, carry_channels=32).eval()
        a = torch.randn(1, 16, 12, 12, requires_grad=True)
        b = torch.randn(1, 32, 12, 12, requires_grad=True)
        carry = torch.randn(1, 32, 12, 12, requires_grad=True)
        result = module(a, b, carry)
        self.assertEqual(tuple(result.shape), (1, 32, 6, 6))
        result.mean().backward()
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in (a, b, carry)))
