import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.egvan.efficientnet_branch import EfficientNetBranch
from models.egvan.modified_resnet50 import ModifiedResNet50
from models.egvan import EGVAN


class PaperAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_branch_taps_match_table_channels_and_stages(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 128, 128)
            efficient = EfficientNetBranch().eval()(x)
            resnet = ModifiedResNet50().eval()(x)
        self.assertEqual([(v.shape[1], *v.shape[-2:]) for v in efficient], [(48, 32, 32), (64, 16, 16), (160, 8, 8), (1280, 4, 4)])
        self.assertEqual([(v.shape[1], *v.shape[-2:]) for v in resnet], [(256, 32, 32), (512, 16, 16), (1024, 8, 8), (2048, 4, 4)])

    def test_figure3_serial_mff_carry_receives_gradient(self):
        model = EGVAN().eval()
        output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 7))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        self.assertIsNotNone(model.fusions[0].pointwise.weight.grad)
        self.assertIsNotNone(model.fusions[1].pointwise.weight.grad)
