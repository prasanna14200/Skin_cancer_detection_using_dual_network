"""Stage 8 synthetic-only architecture and provenance verification."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from models.baseline_effnet import build_model
from models.egvan import EGVAN
from models.egvan.modified_resnet50 import ModifiedResNet50

EXPECTED_CP = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
EXPECTED_SPLIT = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
CP = ROOT / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
SPLIT = ROOT / "data/splits/split_leakage_aware.csv"
PAPER = "https://doi.org/10.1109/ACCESS.2025.3561240"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_csv(name, records, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def count(module):
    return sum(p.numel() for p in module.parameters()), sum(p.numel() for p in module.parameters() if p.requires_grad)

def shape_record(section, scale, tensor):
    return {"section": section, "scale": scale, "batch": tensor.shape[0], "channels": tensor.shape[1], "height": tensor.shape[2], "width": tensor.shape[3]}

def main():
    assert sha(CP) == EXPECTED_CP and sha(SPLIT) == EXPECTED_SPLIT
    torch.manual_seed(8)
    torch.set_num_threads(2)
    baseline, _ = build_model(pretrained=False)
    baseline_count = count(baseline)
    del baseline
    resnet = resnet50(weights=None)
    resnet_count = count(resnet)
    del resnet
    modified = ModifiedResNet50(pretrained=False)
    modified_count = count(modified)
    del modified
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False).eval()
    full_count = count(model)
    module_counts = []
    for name, totals in (("EfficientNetV2S baseline", baseline_count), ("ResNet50 baseline", resnet_count), ("modified ResNet50", modified_count), ("full EG-VAN reconstruction", full_count)):
        module_counts.append({"module": name, "total_parameters": totals[0], "trainable_parameters": totals[1], "method": "sum(p.numel()) on locally instantiated torchvision/PyTorch module; no weights download"})
    for name, module in model.named_children():
        total, trainable = count(module)
        module_counts.append({"module": "EGVAN." + name, "total_parameters": total, "trainable_parameters": trainable, "method": "component count; rows overlap with full total"})
    write_csv("module_parameter_counts.csv", module_counts, ["module", "total_parameters", "trainable_parameters", "method"])

    traces = []
    with torch.inference_mode():
        x = torch.randn(1, 3, 384, 384)
        features = model.forward_features(x)
        logits = model.classifier(model.pool(features["combined"]).flatten(1))
        assert logits.shape == (1, 7) and torch.isfinite(logits).all()
        for section in ("efficient", "resnet", "fused"):
            for scale, tensor in enumerate(features[section], 1):
                assert torch.isfinite(tensor).all()
                traces.append(shape_record(section, scale, tensor))
        traces.append(shape_record("combined", 0, features["combined"]))
        del x, features, logits
        x2 = torch.randn(2, 3, 128, 128)
        y2 = model(x2)
        assert y2.shape == (2, 7) and torch.isfinite(y2).all()
        del x2, y2
    write_csv("feature_shape_trace.csv", traces, ["section", "scale", "batch", "channels", "height", "width"])

    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7) and torch.isfinite(output).all()
    output.mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    trainable_tensor_count = sum(p.requires_grad for p in model.parameters())
    assert len(grads) == trainable_tensor_count and all(torch.isfinite(g).all() for g in grads)
    assert model.efficient.features[0][0].weight.grad is not None
    assert model.resnet.conv1.weight.grad is not None
    assert model.fusions[0].depthwise.weight.grad is not None
    assert model.classifier.weight.grad is not None
    write_json("smoke_test_results.json", {"status": "PASS", "training_performed": False, "dataset_inference_performed": False,
        "pretrained_weights_downloaded": False, "optimizer_step_performed": False,
        "forward_checks": [{"batch": 1, "input_shape": [1, 3, 384, 384], "logits_shape": [1, 7], "finite": True},
                           {"batch": 2, "input_shape": [2, 3, 128, 128], "logits_shape": [2, 7], "finite": True}],
        "backward_check": {"input_shape": [1, 3, 64, 64], "loss": "logits.mean()", "finite_gradients": True,
                           "all_trainable_parameter_tensors_have_gradients": True, "trainable_parameter_tensor_count": trainable_tensor_count,
                           "efficient_stem_gradient": True, "resnet_stem_gradient": True, "mff_gradient": True, "classifier_gradient": True},
        "local_runtime": {"torch": torch.__version__, "device": "cpu"}})

    cross = [
        ("hair removal", "III-A equations 1–7", "src/preprocessing.py", "remove_hair", "EXISTING_APPROXIMATION", "MEDIUM", "Paper top-hat/blackhat ambiguity; no preprocessing changed"),
        ("Gray World", "III-A equations 8–9", "src/preprocessing.py", "gray_world", "EXISTING_APPROXIMATION", "MEDIUM", "Numerical gains/clipping choices documented"),
        ("Retinex", "III-A equations 10–13", "src/preprocessing.py", "retinex", "EXISTING_APPROXIMATION", "MEDIUM", "Paper omits scales/normalization"),
        ("crop", "III-A equation 14", "", "", "UNRESOLVED", "LOW", "No paper crop size/location; not added"),
        ("EfficientNetV2S", "III-C-1; Table 1; Figure 3", "src/models/egvan/efficientnet_branch.py", "EfficientNetBranch", "IMPLEMENTED_WITH_TAP_ASSUMPTION", "MEDIUM", "torchvision stages 2/3/5/7"),
        ("ResNet50", "III-C-1; Table 1; Figure 6", "src/models/egvan/modified_resnet50.py", "ModifiedResNet50", "IMPLEMENTED_WITH_PLACEMENT_ASSUMPTION", "MEDIUM", "SCGA after stages 1/2; NLB after stages 3/4"),
        ("Spatial Attention", "III-C-2; Figure 4", "src/models/egvan/spatial_attention.py", "SpatialAttention", "IMPLEMENTED", "HIGH", "Literal 64/16/8/filter sequence"),
        ("GMA", "III-C-2 equations 16–22", "src/models/egvan/scga.py", "GroupMeanMaxAttention", "IMPLEMENTED_WITH_GROUP_ASSUMPTION", "MEDIUM", "Eight channel groups and directional mean/max"),
        ("SCGA", "III-C-2; Figure 4", "src/models/egvan/scga.py", "SCGA", "IMPLEMENTED", "MEDIUM", "Spatial path followed by GMA"),
        ("Non-Local Block", "III-C-3 equations 23–28; Figure 5", "src/models/egvan/non_local.py", "NonLocalBlock", "IMPLEMENTED_WITH_REDUCTION_ASSUMPTION", "MEDIUM", "Embedded Gaussian; subsampled keys/values"),
        ("MFF", "III-C-5 equations 29–30; Figure 3", "src/models/egvan/mff.py", "MFF", "IMPLEMENTED_WITH_ALIGNMENT_ASSUMPTION", "MEDIUM", "SA/GMA, concatenate, separable stride 2, BN, SCGA"),
        ("GAP", "III-C-6; Figure 3", "src/models/egvan/egvan.py", "EGVAN.pool", "IMPLEMENTED", "HIGH", "Global average pool after aligned fusion"),
        ("classifier", "III-C-6; Figure 3", "src/models/egvan/egvan.py", "EGVAN.classifier", "IMPLEMENTED_WITH_LOGIT_CONVENTION", "HIGH", "Linear logits; softmax external"),
    ]
    write_csv("paper_to_code_crosswalk.csv", [dict(zip(("paper_component", "paper_section_equation_figure", "implementation_file", "implementation_class_or_function", "status", "confidence", "notes"), r)) for r in cross],
              ["paper_component", "paper_section_equation_figure", "implementation_file", "implementation_class_or_function", "status", "confidence", "notes"])
    decision_doc = ROOT / "docs/egvan_implementation_decisions.md"
    assumptions = []
    for line in decision_doc.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("| Status") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("| ").split("|")]
            if len(cells) == 6 and cells[0] in {"EXPLICIT", "INFERRED", "ASSUMED", "UNRESOLVED"}:
                assumptions.append(dict(zip(("status", "paper_statement_and_problem", "chosen_interpretation", "justification", "possible_alternative", "expected_experimental_consequence"), cells)))
    write_csv("implementation_assumptions.csv", assumptions, ["status", "paper_statement_and_problem", "chosen_interpretation", "justification", "possible_alternative", "expected_experimental_consequence"])

    summary = ["EG-VAN Stage 8 reconstruction (synthetic verification only)", f"Paper: {PAPER}", "Architecture: four paired EfficientNetV2S and modified-ResNet50 taps; SA/GMA→separable stride-2 conv→BN→SCGA per scale; aligned fusion→GAP→linear logits", "Verification input size: repository 384×384; paper operational input size not established", "Counts (total / trainable):"]
    summary += [f"  {r['module']}: {r['total_parameters']:,} / {r['trainable_parameters']:,}" for r in module_counts]
    summary += ["MACs/FLOPs: not reported; a full and consistently scoped profiler was unavailable. Paper efficiency equivalence is unverified.", "Synthetic forward: batch 1 at 384² PASS; batch 2 at 128² PASS", "Synthetic backward: batch 1 at 64² PASS; no optimizer step", "Architecture reconstruction != paper's nine-class result reproduction", "No training and no HAM/PH2 dataset inference"]
    (OUT / "architecture_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    report = ["# Stage 8 architecture reconstruction", "", "## Status", "", "**NOT_READY_FOR_TRAINING**. The reconstruction passes synthetic code verification, but paper-faithful training readiness requires visual verification of Figure 3 and Table 1 against the original PDF. Their images were inaccessible: no PDF was found in the accessible repository and the online PDF endpoint returned HTTP 403. The tap choices and exact fusion wiring therefore remain provisional. Supply the PDF path/attachment and review those figures before freezing the architecture.", "", "## Paper authority and topology", "", f"Primary source: [EG-VAN, IEEE Access 2025]({PAPER}). Accessible full text, captions and equations informed the specification; actual Figure 3/Table 1 visuals were not verified. The architecture specification in `docs/egvan_paper_architecture_spec.md` separates explicit statements from inferences and unresolved details. EfficientNetV2S and modified ResNet50 produce four paired scales. The modified ResNet applies SCGA to early stages and embedded-Gaussian Non-Local Blocks to later stages. Each pair enters SA/GMA, concatenation, separable stride-2 convolution, batch normalization and SCGA. Aligned fused outputs enter GAP and a configurable linear classifier.", "", "## Verification", "", "Synthetic forward passed for batch 1 at repository-compatible 384×384 and batch 2 at 128×128; output was finite `[B,7]`. A synthetic batch-1 backward pass propagated finite gradients through all trainable parameter tensors. Component tests cover dimensions and finite gradients. No data images were opened.", "", "## Parameter audit", "", "| Module | Total | Trainable |", "|---|---:|---:|"]
    report += [f"| {r['module']} | {r['total_parameters']:,} | {r['trainable_parameters']:,} |" for r in module_counts]
    report += ["", "These are local instantiated counts. The modified branch and full reconstruction have additional parameters; the paper's efficiency/parameter claims are not adopted. MACs/FLOPs were skipped because a consistently scoped profiler was unavailable without adding dependencies.", "", "## Shape trace", "", "See `feature_shape_trace.csv` for exact 384×384 branch and fusion shapes. Pairwise taps happened to align; final four fused scales required explicit resizing to the coarsest map before concatenation.", "", "## Paper ambiguities and preprocessing", "", "See `docs/egvan_implementation_decisions.md` and `implementation_assumptions.csv`. Major uncertainties are exact tap/attention insertion points, GMA group count, key/value reduction, fusion widths, final alignment, input crop/size and augmentation. The paper's printed bright top-hat equation conflicts with the existing blackhat implementation used for dark hair; the preprocessing pipeline and processed data were left untouched. `src/train.py` uses 384×384, ImageNet normalization and three augmentation operations, none of which is established as paper-exact.", "", "## Research scope", "", "The seven-class forward check is architectural verification, not reproduction of the paper's nine-class reported performance. No training, dataset inference, PH² evaluation, Grad-CAM generation, calibration fitting or optimizer update occurred. Stage 9 must prespecify pretrained initialization, frozen split, input policy and architecture assumptions before training.", "", "## Next action", "", "Obtain and visually inspect the original PDF Figure 3 and Table 1, revise the documented reconstruction if needed, then rerun Stage 8 verification. Only after that: Stage 9 — Controlled EG-VAN Training Protocol.", ""]
    (OUT / "stage8_architecture_report.md").write_text("\n".join(report), encoding="utf-8")

    assert sha(CP) == EXPECTED_CP and sha(SPLIT) == EXPECTED_SPLIT
    sources = [ROOT / "docs/egvan_paper_architecture_spec.md", decision_doc, ROOT / "src/models/baseline_effnet.py", ROOT / "src/preprocessing.py", ROOT / "src/train.py", CP, SPLIT]
    sources += list((ROOT / "src/models/egvan").glob("*.py"))
    sources += [ROOT / "tests" / name for name in ("test_scga.py", "test_non_local.py", "test_mff.py", "test_egvan.py")]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources}
    generated = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "stage8_manifest.json"}
    write_json("stage8_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "paper_doi": PAPER, "paper_figure3_table1_visual_verification": False, "readiness_status": "NOT_READY_FOR_TRAINING", "checkpoint_sha256": EXPECTED_CP, "split_sha256": EXPECTED_SPLIT,
        "source_file_sha256": source_hashes, "generated_file_sha256": generated, "manifest_self_hash_policy": "excluded to avoid recursive hash",
        "training_performed": False, "dataset_inference_performed": False, "gradcam_regenerated": False, "pretrained_weights_downloaded": False,
        "previous_artifacts_modified": False})
    print("STAGE 8 ARCHITECTURE CODE VERIFICATION: PASS; PAPER VISUAL VERIFICATION: PENDING")
    print("Parameter counts:", module_counts[:4])
    print("Forward batch 1 384² / batch 2 128²: PASS; synthetic backward: PASS")
    print("Checkpoint:", sha(CP), "Split:", sha(SPLIT))

if __name__ == "__main__":
    main()
