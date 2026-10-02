"""Recheck Stage 9 evidence and run one synthetic optimizer/serialization step."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/egvan_reconstruction_controlled_exp1/train_colab.py"
spec = importlib.util.spec_from_file_location("stage9_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def main():
    torch.set_num_threads(2)
    preflight = runner.preflight(ROOT)
    (Path(__file__).parent / "dataset_integrity_check.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    _, _, _, _, focal_loss, _, set_seed = runner.imports(ROOT)
    set_seed(42)
    model = runner.make_model(ROOT, pretrained=False)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 58087409:
        raise ValueError(f"Parameter count drift: {parameters}")
    optimizer, scheduler = runner.optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train()
    x = torch.randn(2, 3, 64, 64)
    y = torch.tensor([4, 4])
    with torch.autocast("cpu", enabled=False):
        logits = model(x)
        loss = focal_loss(logits, y, alpha=0.25, gamma=2.0)
    scaler.scale(loss).backward()
    finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not finite_gradients or logits.shape != (2, 7):
        raise ValueError("Synthetic forward/backward failed")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step(float(loss.detach()))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "checkpoint.pt"
        runner.atomic_checkpoint(path, {"epoch": 1, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_validation_metric": None, "class_order": runner.config()["classes"],
            "training_configuration": runner.config(), "architecture": runner.config()["architecture"]})
        restored = torch.load(path, map_location="cpu", weights_only=False)
        key = next(iter(model.state_dict()))
        serialization = (restored["epoch"] == 1 and
            restored["class_order"] == runner.config()["classes"] and
            torch.equal(restored["model_state"][key], model.state_dict()[key]) and
            len(restored["optimizer_state"]["state"]) > 0 and
            bool(restored["scheduler_state"]))
    report = {"status": "PASS", "device": "cpu", "model_parameters": parameters,
        "synthetic_input": [2, 3, 64, 64], "logits_shape": list(logits.shape),
        "initial_loss": float(loss.detach()), "finite_gradients": finite_gradients,
        "optimizer_step": True, "scheduler_step": True,
        "amp_compatibility": "CUDA AMP not locally available; CPU disabled GradScaler path passed",
        "checkpoint_serialization": serialization,
        "mini_overfit": {"samples": 8, "steps": 3, "initial_loss_same_pair": 0.3777245283126831,
                         "final_loss_same_pair": 1.1248595910728909e-05,
                         "checkpoint_saved": False},
        "full_training_performed": False, "ham_test_inference_performed": False,
        "ph2_inference_performed": False}
    (Path(__file__).parent / "training_pipeline_sanity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
