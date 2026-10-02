"""Stage 8B: local-PDF provenance, paper/code audit and synthetic verification."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF = ROOT.parent / "EG-VAN_A_Global_and_Local_Attention-Based_Dual-Branch_Ensemble_Network_With_Advanced_Color_Balancing_for_Multi-Class_Skin_Cancer_Recognition.pdf"
PDF_SHA = "bac1b99167ad820232ecc6b47a3d562d2d9ca55ce19de36fba97b919e8aacee3"
CP = ROOT / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
SPLIT = ROOT / "data/splits/split_leakage_aware.csv"
CP_SHA = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
sys.path.insert(0, str(ROOT / "src"))
from models.baseline_effnet import build_model
from models.egvan import EGVAN
from models.egvan.modified_resnet50 import ModifiedResNet50

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(name, records, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def params(module):
    return sum(p.numel() for p in module.parameters()), sum(p.numel() for p in module.parameters() if p.requires_grad)

def main():
    assert PDF.is_file() and sha(PDF) == PDF_SHA, "Original PDF missing or changed"
    assert sha(CP) == CP_SHA and sha(SPLIT) == SPLIT_SHA, "Frozen artifact hash mismatch"
    protected_dirs = [ROOT / x for x in (
        "experiments/efficientnetv2s_controlled_exp5", "analysis/ph2_external_validation_exp5", "analysis/domain_shift_exp5",
        "analysis/uncertainty_calibration_exp5", "analysis/stage6_failure_mode_synthesis_exp5", "analysis/stage7_final_research_audit_exp5")]
    protected = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for directory in protected_dirs for p in directory.rglob("*") if p.is_file()}
    torch.manual_seed(8)
    torch.set_num_threads(2)

    baseline, _ = build_model(pretrained=False)
    baseline_n = params(baseline)
    del baseline
    resnet = resnet50(weights=None)
    resnet_n = params(resnet)
    resnet_fc_n = params(resnet.fc)
    del resnet
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False).eval()
    modified_n = params(model.resnet)
    full_n = params(model)
    components = [
        ("EfficientNetV2S baseline (7-class)", baseline_n, "Paper backbone family; local baseline head", "Standalone baseline count"),
        ("ResNet50 baseline (1000-class torchvision)", resnet_n, "Paper says modified branch retains ResNet50 parameter count", "Includes unused ImageNet classifier; not directly matched to feature-only branch"),
        ("ResNet50 ImageNet classifier excluded from modified branch", resnet_fc_n, "No classifier in intermediate feature branch", "Subtract before comparing trunk and attention"),
        ("ResNet50 feature trunk in modified branch", (modified_n[0] - sum(params(m)[0] for m in (model.resnet.scga1, model.resnet.scga2, model.resnet.nonlocal3, model.resnet.nonlocal4)),
                                                         modified_n[1] - sum(params(m)[1] for m in (model.resnet.scga1, model.resnet.scga2, model.resnet.nonlocal3, model.resnet.nonlocal4))),
         "Standard ResNet50 feature stages", "Classifier excluded"),
        ("Modified ResNet50 SCGA after layer1", params(model.resnet.scga1), "Table 1 SCGA after 256-channel stage", "Adds trainable spatial/GMA/input-projection weights"),
        ("Modified ResNet50 SCGA after layer2", params(model.resnet.scga2), "Table 1 SCGA after 512-channel stage", "Adds trainable spatial/GMA/input-projection weights"),
        ("Modified ResNet50 Non-Local after layer3", params(model.resnet.nonlocal3), "Table 1 NLB after 1024-channel stage", "Q/K/V and output projections add parameters"),
        ("Modified ResNet50 Non-Local after layer4", params(model.resnet.nonlocal4), "Table 1 NLB after 2048-channel stage", "Q/K/V and output projections add parameters"),
        ("Modified ResNet50 total", modified_n, "Paper claims unchanged ResNet50 count", "Measured count exceeds classifier-free trunk and 1000-class baseline"),
        ("EfficientNetV2S feature branch", params(model.efficient), "Paper pretrained EfficientNetV2S", "Classifier removed; synthetic weights only"),
    ]
    for index, module in enumerate(model.fusions, 1):
        components.append((f"MFF {index}" + (" with prior-scale carry" if index > 1 else ""), params(module), "Figure 3 serial MFF; Section III-C-5 operator order", "Contains spatial, GMA, separable convolution, BN and SCGA"))
    components += [
        ("Final aggregation 1x1", params(model.aggregate), "Figure 3 concatenation before GAP", "Alignment/reduction assumed"),
        ("Seven-class linear head", params(model.classifier), "Paper K-class dense/softmax", "Returns logits; softmax external"),
        ("Full EG-VAN reconstruction", full_n, "Paper claims computational efficiency", "Local count only; not author implementation equivalence"),
    ]
    breakdown = [dict(component=name, parameter_count=n[0], trainable_parameter_count=n[1], paper_expectation=expectation, interpretation=interpretation) for name,n,expectation,interpretation in components]
    write_csv("parameter_count_breakdown.csv", breakdown, list(breakdown[0]))
    attention_added = sum(params(m)[0] for m in (model.resnet.scga1, model.resnet.scga2, model.resnet.nonlocal3, model.resnet.nonlocal4))
    assert modified_n[0] == resnet_n[0] - resnet_fc_n[0] + attention_added
    assert full_n[0] == sum(params(m)[0] for m in (model.efficient, model.resnet, model.fusions, model.aggregate, model.classifier))

    with torch.inference_mode():
        f384 = model.forward_features(torch.randn(1, 3, 384, 384))
        y384 = model.classifier(model.pool(f384["combined"]).flatten(1))
        assert y384.shape == (1, 7) and torch.isfinite(y384).all()
        stage_shapes = [list(t.shape) for t in f384["resnet"]]
        efficient_shapes = [list(t.shape) for t in f384["efficient"]]
        fused_shapes = [list(t.shape) for t in f384["fused"]]
        assert [x[1] for x in stage_shapes] == [256, 512, 1024, 2048]
        assert [x[1] for x in efficient_shapes] == [48, 64, 160, 1280]
        del f384, y384
        y2 = model(torch.randn(2, 3, 128, 128))
        assert y2.shape == (2, 7) and torch.isfinite(y2).all()
        del y2
    y = model(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, 7) and torch.isfinite(y).all()
    y.mean().backward()
    trainable = [(name, p) for name,p in model.named_parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in trainable)
    assert model.fusions[0].pointwise.weight.grad is not None and model.fusions[1].pointwise.weight.grad is not None

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    test_result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert test_result.returncode == 0, test_result.stdout + test_result.stderr
    match = re.search(r"Ran (\d+) tests", test_result.stderr)
    assert match and int(match.group(1)) >= 39
    test_count = int(match.group(1))

    def row(component, reference, description, file, cls, behavior, status, assumption, action):
        return dict(paper_component=component, paper_reference=reference, paper_description=description,
                    implementation_file=file, implementation_class=cls, implementation_behavior=behavior,
                    status=status, assumption=assumption, action_taken=action)
    crosswalk = [
        row("EfficientNetV2S branch", "Table 1; Figure 3", "Stages 0–7 with 24/24/48/64/128/160/256/1280 channels", "src/models/egvan/efficientnet_branch.py", "EfficientNetBranch", "torchvision features; taps 2/3/5/7", "PAPER-CONSISTENT ASSUMPTION", "Tap subset unnumbered in Figure 3", "Verified selected channel dimensions"),
        row("Modified ResNet50", "Table 1; Figure 6", "256/512-channel SCGA stages then 1024/2048-channel NLB stages", "src/models/egvan/modified_resnet50.py", "ModifiedResNet50", "Post-layer1/2 SCGA; post-layer3/4 NLB", "PAPER-CONSISTENT ASSUMPTION", "Post-stage versus inside-bottleneck placement not specified", "Verified channels and order"),
        row("SCGA spatial path", "Figure 4; III-C-2", "64→16(dilation 2)→8→sigmoid mask; GAP refinement; 1x1 input path", "src/models/egvan/spatial_attention.py", "SpatialAttention", "Named convolutions, mask GAP, fixed mask expansion, trainable 1x1 input projection", "PAPER-CONSISTENT ASSUMPTION", "Input 1x1 weight policy not given", "Added Figure 4 input projection"),
        row("Group-wise Mean-Max Attention", "Figure 4; equations 16–22", "Group split; width/height mean/max; directional 1x1 and sigmoid; addition and group concat", "src/models/egvan/scga.py", "GroupMeanMaxAttention", "Eight groups with shared group convolution weights", "PAPER-CONSISTENT ASSUMPTION", "Group count and sharing unspecified", "Verified dimensions and backward"),
        row("SCGA composition", "Figure 4; III-C-2", "Spatial attention feeds GMA", "src/models/egvan/scga.py", "SCGA", "GMA(SpatialAttention(x))", "EXACT", "None for module order", "No correction"),
        row("Non-Local Block", "Figure 5; equations 23–28", "Q/K/V 1x1, embedded Gaussian softmax, value aggregation, projection and residual", "src/models/egvan/non_local.py", "NonLocalBlock", "Embedded Gaussian with stride-2 key/value average pooling", "PAPER-CONSISTENT ASSUMPTION", "Reduction factor and embed width absent", "Verified subsampling and finite gradients"),
        row("MFF per-scale flow", "Figure 3 inset; III-C-5", "Efficient SA + ResNet GMA → concat → separable stride 2 → BN → SCGA", "src/models/egvan/mff.py", "MFF", "Same sequence with depthwise 3x3+pointwise 1x1", "PAPER-CONSISTENT ASSUMPTION", "Separable kernel and output width absent", "No order correction needed"),
        row("Feature extraction points", "Figure 3; Table 1", "Representative early/intermediate/final branch maps; no numbered MFF tap list", "src/models/egvan/efficientnet_branch.py", "tap_indices", "Efficient 2/3/5/7 paired with ResNet 1/2/3/4", "PAPER-UNDERSPECIFIED", "Exact paired stage subset not recoverable", "Retained transparent tap choice"),
        row("Feature-map alignment", "Figure 3; III-C-5", "MFF receives corresponding-scale maps; final maps vary in size", "src/models/egvan/mff.py", "MFF.forward", "Bilinear resize only if paired or carry sizes differ", "PAPER-UNDERSPECIFIED", "Interpolation policy absent", "Retained documented alignment"),
        row("Multi-scale connections", "Figure 3", "Arrows connect successive MFF boxes", "src/models/egvan/egvan.py", "EGVAN.forward_features", "Previous MFF output carried into next MFF concatenation", "PAPER-CONSISTENT ASSUMPTION", "Diagram does not define carry operator", "Corrected missing Stage 8 serial link"),
        row("Final concatenation", "Figure 3; III-C-6", "Last branch maps and MFF outputs feed a final concat node", "src/models/egvan/egvan.py", "EGVAN.forward_features", "Four MFF maps and both terminal branch maps aligned and concatenated", "PAPER-CONSISTENT ASSUMPTION", "Exact alignment and whether every intermediate MFF output enters final concat unspecified", "Corrected missing terminal branch maps"),
        row("Global average pooling", "Figure 3; III-C-6", "Global average pooling after final concatenation", "src/models/egvan/egvan.py", "EGVAN.pool", "GAP after aggregate 1x1 projection", "PAPER-CONSISTENT ASSUMPTION", "Interposed 1x1 projection not explicit", "Retained as dimension reduction"),
        row("Classifier", "Figure 3; III-C-6", "Dense K-way softmax classification", "src/models/egvan/egvan.py", "EGVAN.classifier", "K logits, softmax external", "PAPER-CONSISTENT ASSUMPTION", "PyTorch loss contract", "No correction"),
        row("Output classes", "III-C-6; paper experiments", "K configurable; paper reports nine-class and seven-class experiments", "src/models/egvan/egvan.py", "EGVAN.__init__", "Defaults to seven for frozen project compatibility", "PAPER-CONSISTENT ASSUMPTION", "Stage 8B does not reproduce paper experiments", "No class redesign"),
    ]
    write_csv("paper_code_architecture_crosswalk.csv", crosswalk, list(crosswalk[0]))

    report = ["# Stage 8B final paper-to-code audit", "", "## Status", "", "**READY_FOR_STAGE_9** for protocol design: the visually verifiable Figure 3/Table 1 topology is implemented, the component tests and synthetic passes succeed, and remaining operator choices are declared assumptions. This does not establish author-code identity or reproduce the paper's classification results. Stage 9 training must be separately approved and prespecified.", "", "## Original PDF provenance and visual review", "", f"Local PDF: `{PDF}`; SHA256 `{PDF_SHA}`. Figure 3 and Table 1 were visually inspected on PDF pages 7–8; Figures 4 and 5 on pages 9–10. Figure 6 on page 10. The text and visuals were compared directly with the modules. No data images were used.", "", "## Stage 8B corrections", "", "Stage 8 omitted arrows between MFF boxes and the terminal branch inputs to the final concat. Stage 8B added a previous-MFF carry into the next MFF's concatenation and both terminal branch maps to the final concat. The carry operator and spatial alignment remain assumptions because Figure 3 does not specify their tensor operations. Figure 4's input-side 1×1 convolution was also added.", "", "## Table 1 and stage placement", "", "| ResNet stage | 384×384 output | Paper module | Code module | Audit |", "|---|---|---|---|---|", "| layer1 (paper stage 2) | 256×96×96 | SCGA | SCGA post-stage | Paper-consistent assumption |", "| layer2 (paper stage 3) | 512×48×48 | SCGA | SCGA post-stage | Paper-consistent assumption |", "| layer3 (paper stage 4) | 1024×24×24 | NLB | NLB post-stage | Paper-consistent assumption |", "| layer4 (paper stage 5) | 2048×12×12 | NLB | NLB post-stage | Paper-consistent assumption |", "", "Table 1 confirms the chosen EfficientNet tap channel values 48, 64, 160 and 1280 at stages 2, 3, 5 and 7. The paper does not label their exact MFF connections, so the tap subset remains an assumption.", "", "## SCGA, NLB and MFF", "", "Figure 4 and equations 16–22 support the 64→16(dilation 2)→8→sigmoid mask path, mask GAP refinement, directional mean/max grouping, 1×1 directional projections, addition, sigmoid and group concatenation. Figure 4 also depicts an input-side 1×1 projection, added in Stage 8B. Its trainability is assumed. Figure 5 and equations 23–28 support theta/phi/g projections, embedded-Gaussian affinity, softmax, weighted values, output projection and residual; key/value subsampling is implemented with an assumed factor of two. MFF follows the paper's SA/GMA, concatenation, separable stride-2 convolution, BN and SCGA order. Alignment policy remains assumed.", "", "## Parameter-count discrepancy", "", f"Torchvision ResNet50 with its 1000-class head has {resnet_n[0]:,} parameters. Its feature trunk has {resnet_n[0]-resnet_fc_n[0]:,}. This reconstruction adds {attention_added:,} trainable attention parameters, yielding {modified_n[0]:,} in modified ResNet50. The full reconstruction has {full_n[0]:,}. The largest additions are Q/K/V/output projections in the two NLBs; SCGA contributes smaller convolutional and new input-projection weights. The paper claims unchanged ResNet50 parameter count but does not give enough implementation/accounting detail to reproduce that claim. No layers were removed to force parity. See `parameter_count_breakdown.csv`.", "", "## Tests and synthetic passes", "", f"{test_count} repository tests passed. Batch 1 at 384×384 and batch 2 at 128×128 produced finite `[B,7]` logits. A synthetic 64×64 backward pass produced finite gradients on all {len(trainable)} trainable parameter tensors, including the serial MFF carry. No optimizer step, training, HAM inference or PH² inference occurred.", "", "## Remaining assumptions and next action", "", "Exact EfficientNet tap subset, carry concatenation, eight GMA groups, shared GMA weights, NLB reduction factor, fusion widths, spatial alignment, trainability of Figure 4's input projection and final multi-scale aggregation are not fully specified by the paper. These choices are documented and do not contradict the visible topology. Proceed to Stage 9 protocol design; do not train until that protocol is approved.", ""]
    (OUT / "stage8b_paper_code_report.md").write_text("\n".join(report), encoding="utf-8")
    assert sha(CP) == CP_SHA and sha(SPLIT) == SPLIT_SHA
    assert {path: sha(ROOT / path) for path in protected} == protected, "Protected research artifact changed"
    sources = [PDF, CP, SPLIT, ROOT / "docs/egvan_paper_architecture_spec.md", ROOT / "docs/egvan_implementation_decisions.md"]
    sources += list((ROOT / "src/models/egvan").glob("*.py"))
    sources += list((ROOT / "tests").glob("test_egvan*.py"))
    sources += [ROOT / "tests/test_scga.py", ROOT / "tests/test_mff.py", ROOT / "tests/test_non_local.py"]
    source_hashes = {str(p.relative_to(ROOT.parent)).replace("\\", "/"): sha(p) for p in sources}
    generated = {p.name: sha(p) for p in (OUT / "parameter_count_breakdown.csv", OUT / "paper_code_architecture_crosswalk.csv", OUT / "stage8b_paper_code_report.md", Path(__file__))}
    write_json("stage8b_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "paper_pdf_sha256": PDF_SHA,
        "visual_inspection": {"Figure 3": 7, "Table 1": 8, "Figure 4": 9, "Figure 5": 10, "Figure 6": 10},
        "status": "READY_FOR_STAGE_9", "checkpoint_sha256": CP_SHA, "split_sha256": SPLIT_SHA,
        "source_file_sha256": source_hashes, "generated_file_sha256": generated,
        "protected_artifact_sha256": protected, "training_performed": False, "dataset_inference_performed": False})
    print("STAGE 8B STATUS: READY_FOR_STAGE_9")
    print("Figure 3/Table 1/Figure 4/Figure 5: visually verified")
    print("Paper/code mismatches corrected: serial MFF link, Figure 4 input projection, terminal branch maps in final concatenation")
    print("Tests:", test_count, "passed; synthetic forward/backward: PASS")
    print("Parameter counts:", baseline_n[0], resnet_n[0], modified_n[0], full_n[0])
    print("Frozen checkpoint/split:", sha(CP), sha(SPLIT))

if __name__ == "__main__":
    main()
