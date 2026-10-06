"""Read-only Stage 21 manuscript consistency checks against frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "manuscript/egvan_reconstruction_final.md"
PLOS = ROOT / "manuscript/submission/plos_one/manuscript_source.md"
BACKUPS = ROOT / "analysis/stage21_final_manuscript"
FIGURES = ROOT / "manuscript/submission/plos_one/figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_order(text: str, pattern: str, expected: int) -> None:
    sequence = [int(x) for x in re.findall(pattern, text)]
    if sequence != list(range(1, expected + 1)):
        raise ValueError(f"Numbering failure: {pattern}: {sequence}")


def main() -> None:
    master = MASTER.read_text(encoding="utf-8")
    plos = PLOS.read_text(encoding="utf-8")
    stage16 = ROOT / "analysis/stage16_final_evaluation"
    stage20 = ROOT / "analysis/stage20_reliability_completion"
    ham = json.loads((stage16 / "final_ham_test_metrics.json").read_text(encoding="utf-8"))
    ph2 = json.loads((stage16 / "final_ph2_metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((stage20 / "validation_reliability.json").read_text(encoding="utf-8"))
    reliability = json.loads((stage20 / "reliability_analysis.json").read_text(encoding="utf-8"))
    protocol = json.loads((stage20 / "uncertainty_protocol.json").read_text(encoding="utf-8"))
    ham_reliability = reliability["datasets"]["ham"]
    ph2_reliability = reliability["datasets"]["ph2"]
    op = ham_reliability["frozen_review_rule"]
    if (protocol["metric"] != "entropy" or protocol["threshold"] != 0.7675495327940953 or
        op["accepted"] != 805 or op["reviewed"] != 209 or op["errors_reviewed"] != 106 or
        op["mel_false_negatives_reviewed"] != 23):
        raise ValueError("Frozen Stage 20 review rule/result mismatch")
    if (ham["sample_count"] != 1014 or ham["per_class"]["mel"]["tp"] != 56 or
        ham["per_class"]["mel"]["fn"] != 51 or ph2["sample_count"] != 120 or
        ph2["per_class"]["mel"]["tp"] != 11):
        raise ValueError("Frozen Stage 16 sample/class counts changed")
    if not master.startswith("# EG-VAN+ as a Reliability-Extension Prototype") or not plos.startswith(master.splitlines()[0]):
        raise ValueError("Title mismatch")
    for text, fig_name in ((master, "Figure"), (plos, "Fig")):
        check_order(text, rf"\*\*{fig_name} (\d+)\.", 8)
        check_order(text, r"\*\*Table (\d+)\.", 6)
        citations = {int(value) for value in re.findall(r"\[(\d+)\]", text.split("## References", 1)[0])}
        references = [int(value) for value in re.findall(r"^(\d+)\. ", text.split("## References", 1)[1], re.M)]
        if citations != set(range(1, 8)) or references != list(range(1, 8)):
            raise ValueError(f"Citation/reference mismatch: {citations} {references}")
        required = ("0.7675495327940953", "805/1,014", "209", "106/179", "23/51",
                    "0.019574", "0.029678", "0.060178", "0.270595", "0.256018", "0.491627",
                    "0.827386", "0.858883", "0.766875", "0.823471", "0.699582",
                    "0.958206", "56/107", "0.568528", "11/40", "0.392857")
        missing = [term for term in required if term not in text]
        if missing:
            raise ValueError(f"Missing frozen result strings: {missing}")
        if "90.93% accuracy **among retained cases**" not in text:
            raise ValueError("Abstract does not qualify retained accuracy")
        if "quality gate is incomplete" not in text or "external follow-up" not in text:
            raise ValueError("Missing scope/limitation language")
        source_values = (
            f"{ham['accuracy']:.6f}", f"{ham['macro_f1']:.6f}",
            f"{ham['multiclass_roc_auc']['ovr_macro']:.6f}",
            f"{ham['per_class']['mel']['f1']:.6f}",
            f"{ph2['accuracy']:.6f}", f"{ph2['mel_f1']:.6f}",
            f"{validation['ece_10_bins']:.6f}",
            f"{ham_reliability['calibration']['ece_10_bins']:.6f}",
            f"{ph2_reliability['calibration']['ece_10_bins']:.6f}",
            f"{op['retained_accuracy']:.2%}", f"{op['coverage']:.2%}",
        )
        if any(value not in text for value in source_values):
            raise ValueError(f"Manuscript does not preserve rounded source values: {source_values}")
    refs_master = master.split("## References\n", 1)[1].strip()
    refs_plos = plos.split("## References\n", 1)[1].split("## Supporting information captions", 1)[0].strip()
    if refs_master != refs_plos:
        raise ValueError("Master/PLOS reference lists diverged")
    for label in ("[AUTHOR NAMES", "[AUTHOR AFFILIATIONS", "[NAME, EMAIL", "[AUTHOR TO VERIFY"):
        if label not in plos:
            raise ValueError(f"PLOS author placeholder lost: {label}")
    image_refs = re.findall(r"^!\[[^\]]*\]\(([^)]+)\)", master, re.M)
    if len(image_refs) != 8 or any(not (MASTER.parent / relative).is_file() for relative in image_refs):
        raise ValueError(f"Master figure links missing: {image_refs}")
    sources = {
        5: ROOT / "analysis/stage20_reliability_completion/ham_reliability_diagram.png",
        6: ROOT / "analysis/stage20_reliability_completion/ham_risk_coverage.png",
        7: ROOT / "analysis/stage20_reliability_completion/gradcam_audit_contact_sheet_1.png",
        8: ROOT / "analysis/stage21_final_manuscript/egvan_plus_pipeline.png",
    }
    figure_summary = []
    for number in range(1, 9):
        target = FIGURES / f"Fig{number}.tif"
        if not target.is_file():
            raise FileNotFoundError(target)
        with Image.open(target) as image:
            dpi = image.info.get("dpi", (0, 0))
            if image.width < 1000 or image.height < 1000 or min(dpi) < 300:
                raise ValueError(f"Figure dimensions/DPI: {target.name}: {image.size} {dpi}")
            if number in sources:
                with Image.open(sources[number]) as source:
                    if not np.array_equal(np.asarray(image.convert("RGB")), np.asarray(source.convert("RGB"))):
                        raise ValueError(f"Figure source pixels changed: {target.name}")
            figure_summary.append({"number": number, "dimensions": image.size,
                                   "dpi": tuple(float(value) for value in dpi)})
    for manifest_name in ("final_ham_test_manifest.json", "final_ph2_manifest.json"):
        manifest = json.loads((stage16 / manifest_name).read_text(encoding="utf-8"))
        for artifact, expected in manifest["artifact_sha256"].items():
            if sha256(stage16 / artifact) != expected:
                raise ValueError(f"Stage 16 frozen artifact hash changed: {artifact}")
    backup_master = BACKUPS / "egvan_reconstruction_final.pre_stage21.md"
    backup_plos = BACKUPS / "plos_manuscript_source.pre_stage21.md"
    if sha256(backup_master) != "41f6bfb2c663ab2b5a793f3e8ccd9f04d60ce9f98d323deacdbe6b3d0fb65cad":
        raise ValueError("Master backup changed")
    if sha256(backup_plos) != "0e27006a360f940ce41f9fe6900ebace43129400cc3ab3526494d0bb767b0413":
        raise ValueError("PLOS backup changed")
    print(json.dumps({"status": "PASS", "master_sha256": sha256(MASTER), "plos_sha256": sha256(PLOS),
                      "tables": 6, "figures": figure_summary, "references": 7,
                      "master_backup_sha256": sha256(backup_master),
                      "plos_backup_sha256": sha256(backup_plos)}))


if __name__ == "__main__":
    main()
