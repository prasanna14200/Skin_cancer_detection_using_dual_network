"""Colab entry point for the bounded, train-only Stage 13B A/B diagnostic."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/egvan_melanoma_ablation_exp/diagnose_first_epoch.py"
if not RUNNER.is_file():
    raise FileNotFoundError(RUNNER)
sys.path.insert(0, str(RUNNER.parent))
runpy.run_path(str(RUNNER), run_name="__main__")
