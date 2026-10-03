"""Stage 12 saved-artifact comparison. No checkpoint loading or inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
EXPECTED = {
    "egvan_checkpoint": ("experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt", "60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0"),
    "experiment5_checkpoint": ("experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt", "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"),
    "frozen_split": ("data/splits/split_leakage_aware.csv", "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"),
    "egvan_ham_predictions": ("analysis/egvan_internal_test_exp1/test_predictions.csv", "c290aebcaba0309c4e7a0ea472d7628593772a1587719c8a02e59f6d33a6ec69"),
    "egvan_ham_metrics": ("analysis/egvan_internal_test_exp1/test_metrics.json", "8cba58f7eadd5722bdc479f8063c57422673984f74a403b4f521868171649e3e"),
    "experiment5_ham_predictions": ("experiments/efficientnetv2s_controlled_exp5/test_predictions.csv", "a33db393a50dd3184ac387724aca4299ac6945fa6c71655c8e54bc9a73ccef06"),
    "experiment5_ham_metrics": ("experiments/efficientnetv2s_controlled_exp5/test_metrics.json", "b3632a5fa736ad4057a9ecd2f601bd4fbc5c3611fb827c2a2231be5a152711db"),
    "egvan_ph2_predictions": ("analysis/egvan_ph2_external_followup_exp1/ph2_predictions.csv", "87e4ae44fd8721ba87d098008254d1946673f2a37aae8d5dc5f0a9bcd98052b6"),
    "egvan_ph2_metrics": ("analysis/egvan_ph2_external_followup_exp1/ph2_metrics.json", "232ce91c5d85d210c56489202a2437250a68290d71833a5c4bb49c3853b92141"),
    "experiment5_ph2_predictions": ("analysis/ph2_external_validation_exp5/ph2_predictions.csv", "7663e0eb8a2fea96873005a4780f0008bcf9b81ae3d74ce413b197912aa9c5bc"),
    "experiment5_ph2_metrics": ("analysis/ph2_external_validation_exp5/ph2_metrics.json", "08e6d586a0fc82a362a6ec5060ce317f2b82c62b665307116cc64257a1e2091f"),
    "ph2_cohort_manifest": ("analysis/ph2_external_validation_exp5/ph2_manifest.csv", "3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842"),
}
OUTPUTS = ("stage12_comparison_report.md", "ham_model_comparison.csv", "ph2_model_comparison.csv",
           "ham_paired_outcomes.csv", "ph2_paired_outcomes.csv", "melanoma_failure_comparison.csv",
           "external_degradation_comparison.csv", "uncertainty_comparison.csv", "improvement_hypotheses.csv",
           "claim_evidence_matrix.csv", "stage12_summary.json", "stage12_experiment_manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def assert_close(a, b, name: str, tolerance=1e-12):
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name} mismatch: {a} vs {b}")


def load_inputs() -> dict:
    if any((OUT / name).exists() for name in OUTPUTS):
        raise FileExistsError("Stage 12 output already exists; refusing overwrite")
    missing = [name for name, (relative, _) in EXPECTED.items() if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 10/11 or control artifacts missing: {missing}")
    hashes = {name: sha256(ROOT / relative) for name, (relative, _) in EXPECTED.items()}
    wrong = {name: value for name, value in hashes.items() if value != EXPECTED[name][1]}
    if wrong:
        raise ValueError(f"Frozen source hash mismatch: {wrong}")
    data = {"hashes": hashes}
    for key in ("experiment5_ham", "egvan_ham", "experiment5_ph2", "egvan_ph2"):
        data[key] = {"rows": read_csv(ROOT / EXPECTED[f"{key}_predictions"][0]),
                     "saved": json.loads((ROOT / EXPECTED[f"{key}_metrics"][0]).read_text(encoding="utf-8"))}
    data["split"] = read_csv(ROOT / EXPECTED["frozen_split"][0])
    data["ph2_cohort"] = read_csv(ROOT / EXPECTED["ph2_cohort_manifest"][0])
    stage10 = json.loads((ROOT / "analysis/egvan_internal_test_exp1/stage10_manifest.json").read_text())
    stage11 = json.loads((ROOT / "analysis/egvan_ph2_external_followup_exp1/stage11_manifest.json").read_text())
    if (stage10.get("status") != "COMPLETE" or stage11.get("status") != "COMPLETE" or
            stage10["generated_artifact_hashes"]["test_predictions.csv"] != hashes["egvan_ham_predictions"] or
            stage11["generated_artifact_hashes"]["ph2_predictions.csv"] != hashes["egvan_ph2_predictions"] or
            stage10["prediction_integrity"]["status"] != "PASS" or stage11["prediction_integrity"]["status"] != "PASS"):
        raise ValueError("Stage 10/11 completion or provenance mismatch")
    return data


def make_records(rows: list[dict], dataset: str, expected: dict[str, str]) -> list[dict]:
    if len(rows) != len(expected) or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError(f"{dataset} row count or duplicate ID mismatch")
    if {r["image_id"] for r in rows} != set(expected):
        raise ValueError(f"{dataset} IDs differ from frozen cohort")
    records = []
    for row in rows:
        image_id = row["image_id"]
        if dataset == "HAM":
            true, pred = row["true_label"], row["predicted_label"]
            probs = [float(x) for x in json.loads(row["probabilities"])]
            if (row["correct"].strip().lower() == "true") != (true == pred):
                raise ValueError(f"Incorrect HAM correctness flag: {image_id}")
        else:
            true, pred = row["true_ham_label"], row["predicted_class"]
            probs = [float(row[f"p_{name}"]) for name in CLASSES]
            if row["true_external_label"] != ("common nevus" if true == "nv" else "melanoma"):
                raise ValueError(f"Incorrect PH2 external label: {image_id}")
        if true != expected[image_id] or true not in CLASSES or pred not in CLASSES:
            raise ValueError(f"True/predicted class mismatch: {image_id}")
        if (len(probs) != 7 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
                or abs(sum(probs) - 1) > 1e-5):
            raise ValueError(f"Invalid seven-class probabilities: {image_id}")
        if CLASSES[max(range(7), key=lambda i: probs[i])] != pred:
            raise ValueError(f"Probability argmax mismatch: {image_id}")
        confidence = max(probs)
        records.append({"image_id": image_id, "true": true, "pred": pred, "correct": true == pred,
                        "probabilities": probs, "confidence": confidence,
                        "entropy": -sum(p * math.log(max(p, 1e-12)) for p in probs),
                        "true_probability": probs[CLASSES.index(true)]})
    return records


def ham_metrics(records: list[dict]) -> dict:
    matrix = [[0] * 7 for _ in range(7)]
    for row in records:
        matrix[CLASSES.index(row["true"])][CLASSES.index(row["pred"])] += 1
    per = {}
    for i, name in enumerate(CLASSES):
        support = sum(matrix[i]); predicted = sum(r[i] for r in matrix); tp = matrix[i][i]
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1,
                     "predicted_count": predicted}
    n = len(records)
    return {"sample_count": n, "accuracy": sum(matrix[i][i] for i in range(7))/n,
            "balanced_accuracy": sum(v["recall"] for v in per.values())/7,
            "macro_precision": sum(v["precision"] for v in per.values())/7,
            "macro_recall": sum(v["recall"] for v in per.values())/7,
            "macro_f1": sum(v["f1"] for v in per.values())/7,
            "weighted_f1": sum(v["f1"]*v["support"] for v in per.values())/n,
            "per_class": per, "confusion_matrix": matrix}


def ph2_metrics(records: list[dict]) -> dict:
    m = [[0, 0, 0], [0, 0, 0]]
    for row in records:
        i = 0 if row["true"] == "nv" else 1
        j = 0 if row["pred"] == "nv" else 1 if row["pred"] == "mel" else 2
        m[i][j] += 1
    per = {}
    for i, name in enumerate(("nv", "mel")):
        support = sum(m[i]); predicted = m[0][i]+m[1][i]; tp=m[i][i]
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {"sample_count": len(records), "accuracy": (m[0][0]+m[1][1])/len(records),
            "balanced_accuracy": (per["nv"]["recall"]+per["mel"]["recall"])/2,
            "per_class": per, "confusion_matrix": m,
            "prediction_distribution": dict(Counter(r["pred"] for r in records))}


def compare_saved(derived: dict, saved: dict, dataset: str):
    if dataset == "HAM":
        if derived["confusion_matrix"] != saved["confusion_matrix"] or derived["sample_count"] != saved["sample_count"]:
            raise ValueError("Saved HAM confusion matrix/count mismatch")
        if tuple(saved["class_order"]) != CLASSES:
            raise ValueError("Saved HAM class order mismatch")
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert_close(derived[key], saved[key], f"HAM {key}")
        for name in CLASSES:
            for key in ("support", "precision", "recall", "f1"):
                assert_close(derived["per_class"][name][key], saved["per_class"][name][key], f"HAM {name} {key}")
    else:
        if derived["confusion_matrix"] != saved["confusion_matrix"]["counts"] or derived["sample_count"] != saved["total_samples"]:
            raise ValueError("Saved PH2 confusion matrix/count mismatch")
        for key in ("accuracy", "balanced_accuracy"):
            assert_close(derived[key], saved[key], f"PH2 {key}")
        for name in ("nv", "mel"):
            for key in ("precision", "recall", "f1"):
                assert_close(derived["per_class"][name][key], saved["class_metrics"][name][key], f"PH2 {name} {key}")


def paired(a: list[dict], b: list[dict], dataset: str) -> tuple[list[dict], dict]:
    baseline = {r["image_id"]: r for r in a}
    egvan = {r["image_id"]: r for r in b}
    if set(baseline) != set(egvan):
        raise ValueError(f"{dataset} paired IDs mismatch")
    counts = Counter()
    rows = []
    for image_id in sorted(baseline):
        old, new = baseline[image_id], egvan[image_id]
        if old["true"] != new["true"]:
            raise ValueError(f"{dataset} paired true class mismatch")
        outcome = ("both_correct" if old["correct"] and new["correct"] else
                   "egvan_only_correct" if new["correct"] else
                   "experiment5_only_correct" if old["correct"] else "both_wrong")
        counts[outcome] += 1
        rows.append({"image_id": image_id, "true_label": old["true"],
                     "experiment5_predicted": old["pred"], "egvan_predicted": new["pred"],
                     "experiment5_correct": old["correct"], "egvan_correct": new["correct"], "outcome": outcome})
    return rows, {key: counts[key] for key in ("both_correct", "egvan_only_correct", "experiment5_only_correct", "both_wrong")}


def mcnemar_exact(b: int, c: int) -> float:
    n = b+c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c)+1)) / (2**n))


def uncertainty(records: list[dict], dataset: str, model: str) -> dict:
    n = len(records)
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]
    mean = lambda values: sum(values)/len(values) if values else None
    bins = [[] for _ in range(10)]
    for row in records:
        bins[min(int(row["confidence"]*10), 9)].append(row)
    ece = sum((len(group)/n)*abs(mean([r["correct"] for r in group]) - mean([r["confidence"] for r in group]))
              for group in bins if group)
    brier = mean([sum((r["probabilities"][i]-(1.0 if CLASSES[i]==r["true"] else 0.0))**2
                      for i in range(7)) for r in records])
    return {"dataset": dataset, "model": model, "sample_count": n, "correct_count": len(correct),
            "incorrect_count": len(wrong), "mean_confidence": mean([r["confidence"] for r in records]),
            "mean_confidence_correct": mean([r["confidence"] for r in correct]),
            "mean_confidence_incorrect": mean([r["confidence"] for r in wrong]),
            "mean_entropy": mean([r["entropy"] for r in records]),
            "mean_entropy_correct": mean([r["entropy"] for r in correct]),
            "mean_entropy_incorrect": mean([r["entropy"] for r in wrong]),
            "ece_10_bins": ece, "multiclass_brier": brier,
            "nll": -mean([math.log(max(r["true_probability"],1e-12)) for r in records]),
            "high_confidence_errors_ge_0_75": sum(not r["correct"] and r["confidence"] >= 0.75 for r in records)}


def metric_rows(base: dict, eg: dict, dataset: str) -> list[dict]:
    keys = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1") if dataset == "HAM" else ("accuracy", "balanced_accuracy")
    rows = [{"metric": key, "scope": "overall", "experiment5": base[key], "egvan": eg[key],
             "egvan_minus_experiment5": eg[key]-base[key]} for key in keys]
    names = CLASSES if dataset == "HAM" else ("nv", "mel")
    for name in names:
        for key in ("precision", "recall", "f1", "support"):
            a,b=base["per_class"][name][key],eg["per_class"][name][key]
            rows.append({"metric": key, "scope": name, "experiment5": a, "egvan": b,
                         "egvan_minus_experiment5": b-a})
    return rows


def degradation_rows(data: dict) -> list[dict]:
    rows=[]
    for model, prefix in (("EfficientNetV2S Experiment #5","experiment5"),("Reconstructed EG-VAN","egvan")):
        ham,ph2=data[f"{prefix}_ham"]["derived"],data[f"{prefix}_ph2"]["derived"]
        for metric,ham_value,ph_value in (("accuracy",ham["accuracy"],ph2["accuracy"]),
             ("melanoma_recall",ham["per_class"]["mel"]["recall"],ph2["per_class"]["mel"]["recall"]),
             ("nevus_recall",ham["per_class"]["nv"]["recall"],ph2["per_class"]["nv"]["recall"])):
            rows.append({"model":model,"metric":metric,"ham":ham_value,"ph2":ph_value,
                         "ph2_minus_ham":ph_value-ham_value,"absolute_drop":ham_value-ph_value})
    return rows


def hypothesis_rows(data: dict) -> list[dict]:
    return [
      {"priority":1,"observed_problem":"EG-VAN melanoma recall trails baseline on HAM and PH2",
       "supporting_evidence":"HAM 56/107 vs 59/107; PH2 11/40 vs 12/40; paired MEL outcomes in saved tables",
       "possible_experiment":"Pre-register HAM train/validation-only ablation of current sampler and focal objective; freeze selection rule before any new external check",
       "requires_training":"yes","uses_ph2_for_tuning":"no; PH2 descriptive only",
       "scientific_risk":"HAM test has already been viewed; repeated test-driven iteration would bias claims",
       "recommendation":"A: exploratory HAM-only design; C: confirm selected model on a new untouched external cohort"},
      {"priority":2,"observed_problem":"Modified ResNet50 branch was randomly initialized while EfficientNet branch used ImageNet weights",
       "supporting_evidence":"Stage 9 frozen config and Stage 8B decision log; not a causal explanation of observed errors",
       "possible_experiment":"Pre-register pretrained-versus-random ResNet branch ablation with otherwise fixed architecture and HAM-only selection",
       "requires_training":"yes","uses_ph2_for_tuning":"no",
       "scientific_risk":"Architecture/weight-source ambiguity; multiple comparisons and extra compute",
       "recommendation":"A: controlled architectural ablation; C: untouched external confirmation"},
      {"priority":3,"observed_problem":"External melanoma recall remains low in both models",
       "supporting_evidence":"Experiment #5 12/40 and EG-VAN 11/40 true MEL correct on mapped PH2",
       "possible_experiment":"Acquire a new untouched external dermoscopy cohort for locked-model confirmation",
       "requires_training":"no","uses_ph2_for_tuning":"no",
       "scientific_risk":"PH2 had prior diagnostic use and only 40 melanoma cases",
       "recommendation":"C: prioritize independent confirmation before generalization claims"},
      {"priority":4,"observed_problem":"Prediction confidence and errors differ across datasets",
       "supporting_evidence":"Saved-probability ECE, Brier, NLL and high-confidence-error rows in uncertainty_comparison.csv",
       "possible_experiment":"Assess calibration methods on HAM validation only; evaluate once on a new untouched external cohort",
       "requires_training":"no for post-hoc fit; yes for new model variants","uses_ph2_for_tuning":"no",
       "scientific_risk":"Fitting temperature or thresholds to PH2 would leak the follow-up cohort",
       "recommendation":"A/C: future pre-registered calibration study; no calibration in Stage 12"},
      {"priority":5,"observed_problem":"PH2 nevus recall differs between models but PH2 composition differs from HAM",
       "supporting_evidence":"PH2 NV 70/80 EG-VAN vs 63/80 baseline; seven-class HAM vs mapped two-class PH2",
       "possible_experiment":"Do not optimize augmentation, preprocessing, or thresholds against PH2 outcomes",
       "requires_training":"no","uses_ph2_for_tuning":"yes if chosen from these outcomes",
       "scientific_risk":"Improper external-cohort tuning and misleading domain-shift attribution",
       "recommendation":"B: reject PH2-driven model selection; C: use untouched external cohort for final test"}
    ]


def claim_rows(data: dict) -> list[dict]:
    return [
      {"claim":"EG-VAN had higher PH2 mapped-cohort accuracy","supported":"descriptive yes",
       "evidence":"81/120 vs 75/120; paired b=13 c=7","limitation":"Mapped two-class cohort with prior PH2 exposure; exact McNemar p=0.2632",
       "safe_paper_wording":"On the mapped PH2 follow-up cohort, EG-VAN accuracy was 0.675 versus 0.625 for the baseline."},
      {"claim":"EG-VAN is statistically superior","supported":"no",
       "evidence":"Exact PH2 paired McNemar p=0.26317596435546875",
       "limitation":"No statistically supported superiority; small exposed cohort",
       "safe_paper_wording":"The observed paired PH2 accuracy difference did not establish superiority."},
      {"claim":"EG-VAN improves melanoma detection","supported":"no",
       "evidence":"HAM MEL recall 56/107 vs 59/107; PH2 MEL recall 11/40 vs 12/40",
       "limitation":"Both are lower for EG-VAN; precision and F1 may move differently",
       "safe_paper_wording":"EG-VAN did not improve melanoma recall in either saved evaluation."},
      {"claim":"EG-VAN had a smaller HAM-to-PH2 accuracy decrease","supported":"descriptive yes",
       "evidence":"Drops 0.14946 EG-VAN vs 0.20833 baseline",
       "limitation":"Seven-class HAM and mapped NV/MEL PH2 populations differ; not causal domain-shift magnitude",
       "safe_paper_wording":"The descriptive accuracy difference between HAM and PH2 was smaller for EG-VAN in these cohorts."},
      {"claim":"EG-VAN had a smaller HAM-to-PH2 nevus recall decrease","supported":"descriptive yes",
       "evidence":"Drops 0.06435 EG-VAN vs 0.14445 baseline",
       "limitation":"Different datasets and class compositions; no causal attribution",
       "safe_paper_wording":"The descriptive NV recall decrease was smaller for EG-VAN."},
      {"claim":"EG-VAN generalizes clinically","supported":"no",
       "evidence":"HAM and one prior-exposed PH2 follow-up only",
       "limitation":"No untouched external confirmation or clinical deployment study",
       "safe_paper_wording":"These results motivate further evaluation on an untouched external cohort."}
    ]


def main():
    data=load_inputs()
    split_test={r["image_id"]:r["dx"] for r in data["split"] if r["split"]=="test"}
    if len(split_test)!=1014 or Counter(split_test.values()) != {"akiec":40,"bcc":58,"bkl":104,"df":11,"mel":107,"nv":676,"vasc":18}:
        raise ValueError("Frozen HAM test cohort mismatch")
    ph2_included={r["image_id"]:r["true_ham_label"] for r in data["ph2_cohort"] if r["included"].lower()=="true"}
    if (len(data["ph2_cohort"])!=200 or len(ph2_included)!=120 or
            Counter(ph2_included.values())!={"nv":80,"mel":40} or
            sum(r["included"].lower()=="false" and r["true_external_label"]=="atypical nevus" for r in data["ph2_cohort"])!=80):
        raise ValueError("Frozen PH2 mapped cohort mismatch")
    for key in ("experiment5_ham","egvan_ham","experiment5_ph2","egvan_ph2"):
        dataset="HAM" if key.endswith("ham") else "PH2"
        data[key]["records"]=make_records(data[key]["rows"],dataset,split_test if dataset=="HAM" else ph2_included)
        data[key]["derived"]=ham_metrics(data[key]["records"]) if dataset=="HAM" else ph2_metrics(data[key]["records"])
        compare_saved(data[key]["derived"],data[key]["saved"],dataset)
    ham_pair,ham_counts=paired(data["experiment5_ham"]["records"],data["egvan_ham"]["records"],"HAM")
    ph2_pair,ph2_counts=paired(data["experiment5_ph2"]["records"],data["egvan_ph2"]["records"],"PH2")
    if ph2_counts != {"both_correct":68,"egvan_only_correct":13,"experiment5_only_correct":7,"both_wrong":32}:
        raise ValueError(f"PH2 paired counts differ from Stage 11: {ph2_counts}")
    ph2_p=mcnemar_exact(ph2_counts["egvan_only_correct"],ph2_counts["experiment5_only_correct"])
    assert_close(ph2_p,0.26317596435546875,"PH2 McNemar exact")
    melanoma=[]
    for dataset, pair, old, new in (("HAM",ham_pair,data["experiment5_ham"]["records"],data["egvan_ham"]["records"]),
                                    ("PH2",ph2_pair,data["experiment5_ph2"]["records"],data["egvan_ph2"]["records"])):
        old_by={r["image_id"]:r for r in old}; new_by={r["image_id"]:r for r in new}
        for row in pair:
            if row["true_label"]=="mel":
                a,b=old_by[row["image_id"]],new_by[row["image_id"]]
                melanoma.append({"dataset":dataset,"image_id":row["image_id"],"true_label":"mel",
                                 "experiment5_predicted":a["pred"],"egvan_predicted":b["pred"],
                                 "experiment5_correct":a["correct"],"egvan_correct":b["correct"],
                                 "outcome":row["outcome"],"experiment5_confidence":a["confidence"],
                                 "egvan_confidence":b["confidence"]})
    unc=[]
    for dataset,key in (("HAM","ham"),("PH2","ph2")):
        for model,prefix in (("EfficientNetV2S Experiment #5","experiment5"),("Reconstructed EG-VAN","egvan")):
            unc.append(uncertainty(data[f"{prefix}_{key}"]["records"],dataset,model))
    # The project's existing saved-probability implementation is independently
    # reproduced for the two Experiment #5 rows before writing Stage 12 outputs.
    cal=json.loads((ROOT/"analysis/uncertainty_calibration_exp5/calibration_metrics.json").read_text())
    for row,label in ((unc[0],"HAM10000"),(unc[2],"PH2")):
        for ours,theirs in (("ece_10_bins","ece"),("multiclass_brier","brier_score"),("nll","nll"),
                             ("mean_confidence","mean_confidence"),("mean_entropy","mean_entropy")):
            assert_close(row[ours],cal[label][theirs],f"baseline calibration {label}/{ours}",1e-11)
    shifts=degradation_rows(data)
    ham_rows=metric_rows(data["experiment5_ham"]["derived"],data["egvan_ham"]["derived"],"HAM")
    ph2_rows=metric_rows(data["experiment5_ph2"]["derived"],data["egvan_ph2"]["derived"],"PH2")
    hypotheses=hypothesis_rows(data); claims=claim_rows(data)
    write_csv(OUT/"ham_model_comparison.csv",ham_rows,list(ham_rows[0]))
    write_csv(OUT/"ph2_model_comparison.csv",ph2_rows,list(ph2_rows[0]))
    write_csv(OUT/"ham_paired_outcomes.csv",ham_pair,list(ham_pair[0]))
    write_csv(OUT/"ph2_paired_outcomes.csv",ph2_pair,list(ph2_pair[0]))
    write_csv(OUT/"melanoma_failure_comparison.csv",melanoma,list(melanoma[0]))
    write_csv(OUT/"external_degradation_comparison.csv",shifts,list(shifts[0]))
    write_csv(OUT/"uncertainty_comparison.csv",unc,list(unc[0]))
    write_csv(OUT/"improvement_hypotheses.csv",hypotheses,list(hypotheses[0]))
    write_csv(OUT/"claim_evidence_matrix.csv",claims,list(claims[0]))
    by_ds={ds:Counter(r["outcome"] for r in melanoma if r["dataset"]==ds) for ds in ("HAM","PH2")}
    ph2_mel={model:Counter(r["pred"] for r in data[f"{model}_ph2"]["records"] if r["true"]=="mel")
             for model in ("experiment5","egvan")}
    ph2_nv={model:Counter(r["pred"] for r in data[f"{model}_ph2"]["records"] if r["true"]=="nv")
            for model in ("experiment5","egvan")}
    summary={"status":"PASS","class_order":list(CLASSES),"source_hashes":data["hashes"],
             "ham_cases":1014,"ph2_cases":120,"ph2_composition":{"nv":80,"mel":40},
             "ham_paired":ham_counts,"ph2_paired":ph2_counts,
             "mcnemar_exact":{"ham_b":ham_counts["egvan_only_correct"],"ham_c":ham_counts["experiment5_only_correct"],
                              "ham_p_value":mcnemar_exact(ham_counts["egvan_only_correct"],ham_counts["experiment5_only_correct"]),
                              "ph2_b":13,"ph2_c":7,"ph2_p_value":ph2_p},
             "melanoma_paired":{"HAM":dict(by_ds["HAM"]),"PH2":dict(by_ds["PH2"])},
             "ph2_melanoma_predictions":{k:dict(v) for k,v in ph2_mel.items()},
             "ph2_nevus_predictions":{k:dict(v) for k,v in ph2_nv.items()},
             "model_metrics":{k:data[k]["derived"] for k in ("experiment5_ham","egvan_ham","experiment5_ph2","egvan_ph2")},
             "training_performed":False,"inference_performed":False,"checkpoint_modified":False,
             "ph2_limitation":"External follow-up with prior project diagnostic/evaluation exposure; not untouched independent validation."}
    write_json(OUT/"stage12_summary.json",summary)
    report_text=report(summary,ham_rows,ph2_rows,shifts,unc)
    with (OUT/"stage12_comparison_report.md").open("x",encoding="utf-8") as f:
        f.write(report_text)
    # A final source rehash detects any concurrent changes during generation.
    after={name:sha256(ROOT/relative) for name,(relative,_) in EXPECTED.items()}
    if after!=data["hashes"]:
        raise ValueError("Frozen source hash changed while generating Stage 12")
    manifest={"status":"PASS","analysis_only":True,"training_performed":False,"inference_performed":False,
              "checkpoint_modified":False,"source_artifact_hashes":after,
              "generated_artifact_hashes":{name:sha256(OUT/name) for name in OUTPUTS if name!="stage12_experiment_manifest.json"},
              "rows":{"ham_paired":len(ham_pair),"ph2_paired":len(ph2_pair),"melanoma":len(melanoma)},
              "class_order":list(CLASSES),"ph2_limitation":summary["ph2_limitation"]}
    write_json(OUT/"stage12_experiment_manifest.json",manifest)
    print(json.dumps({"status":"PASS","ham_paired":ham_counts,"ph2_paired":ph2_counts,
                      "ph2_mcnemar_exact_p":ph2_p,"ph2_melanoma":summary["ph2_melanoma_predictions"],
                      "source_hashes_unchanged":True},indent=2))


def report(summary,ham_rows,ph2_rows,shifts,unc):
    m=summary["model_metrics"]
    hb,he,pb,pe=(m[k] for k in ("experiment5_ham","egvan_ham","experiment5_ph2","egvan_ph2"))
    fmt=lambda x:f"{x:.6f}"
    lines=["# Stage 12: frozen baseline versus reconstructed EG-VAN", "",
           "This is analysis of saved predictions only. Both checkpoints, the HAM split, and Stage 10/11 outputs were hash-verified. No training, inference, threshold tuning, calibration fitting, or Grad-CAM was performed. PH² is an external follow-up with prior project exposure, not untouched independent validation.","",
           "## HAM internal test (1,014 paired cases)","",
           "| Metric | Experiment #5 | EG-VAN | EG-VAN minus baseline |","|---|---:|---:|---:|"]
    for r in ham_rows[:6]: lines.append(f"| {r['metric']} | {fmt(r['experiment5'])} | {fmt(r['egvan'])} | {r['egvan_minus_experiment5']:+.6f} |")
    lines += ["",f"Paired correctness: {summary['ham_paired']}. Exact McNemar b={summary['mcnemar_exact']['ham_b']}, c={summary['mcnemar_exact']['ham_c']}, p={summary['mcnemar_exact']['ham_p_value']:.6f}. This is a paired accuracy test, not evidence about all class-specific metrics.","",
              f"Melanoma recall: baseline {hb['per_class']['mel']['recall']:.6f} ({round(hb['per_class']['mel']['recall']*107)}/107), EG-VAN {he['per_class']['mel']['recall']:.6f} ({round(he['per_class']['mel']['recall']*107)}/107). NV recall: baseline {hb['per_class']['nv']['recall']:.6f}, EG-VAN {he['per_class']['nv']['recall']:.6f}. Per-class precision, recall, F1 and support are in `ham_model_comparison.csv`.","",
              "## PH² mapped external follow-up (120 paired cases; 80 NV, 40 MEL)","",
              "| Metric | Experiment #5 | EG-VAN | EG-VAN minus baseline |","|---|---:|---:|---:|"]
    for r in ph2_rows[:2]: lines.append(f"| {r['metric']} | {fmt(r['experiment5'])} | {fmt(r['egvan'])} | {r['egvan_minus_experiment5']:+.6f} |")
    lines += ["",f"MEL recall: {pb['per_class']['mel']['recall']:.3f} (12/40) versus {pe['per_class']['mel']['recall']:.3f} (11/40). NV recall: {pb['per_class']['nv']['recall']:.4f} (63/80) versus {pe['per_class']['nv']['recall']:.4f} (70/80). The six additional correct EG-VAN predictions overall come from a net seven extra correct NV and one fewer correct MEL case; this is a class-level accounting, not a causal explanation.","",
              f"Paired correctness: {summary['ph2_paired']}; exact McNemar b=13, c=7, p={summary['mcnemar_exact']['ph2_p_value']:.6f}. This does not support a statistical superiority claim.","",
              f"True MEL prediction distribution: baseline {summary['ph2_melanoma_predictions']['experiment5']}; EG-VAN {summary['ph2_melanoma_predictions']['egvan']}. True NV distribution: baseline {summary['ph2_nevus_predictions']['experiment5']}; EG-VAN {summary['ph2_nevus_predictions']['egvan']}. Predictions into other HAM classes remain errors.","",
              f"MEL paired cases: HAM {summary['melanoma_paired']['HAM']}; PH² {summary['melanoma_paired']['PH2']}. The case-level transitions are in `melanoma_failure_comparison.csv`.","",
              "## Confidence and uncertainty from saved probabilities", "",
              "ECE uses the project's 10 fixed equal-width top-class confidence bins; Brier is the mean seven-class sum of squared errors; NLL uses natural log with 1e-12 floor; high-confidence error means confidence ≥0.75. No calibration was fitted.","",
              "| Dataset | Model | Mean confidence | Mean entropy | ECE | Brier | NLL | High-confidence errors |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in unc:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['mean_confidence']:.4f} | {row['mean_entropy']:.4f} | {row['ece_10_bins']:.4f} | {row['multiclass_brier']:.4f} | {row['nll']:.4f} | {row['high_confidence_errors_ge_0_75']} |")
    lines += ["","Confidence and entropy split by correct/incorrect predictions are in `uncertainty_comparison.csv`. ECE, Brier and NLL are descriptive on these cohorts; PH² has only NV/MEL truth but all seven predicted classes are retained.","",
              "## HAM to PH² descriptive changes", "",
              "| Model | Metric | HAM | PH² | PH² minus HAM |","|---|---|---:|---:|---:|"]
    for r in shifts:
        lines.append(f"| {r['model']} | {r['metric']} | {r['ham']:.6f} | {r['ph2']:.6f} | {r['ph2_minus_ham']:+.6f} |")
    lines += ["","The accuracy decrease was numerically smaller for EG-VAN, as was the NV recall decrease. MEL recall remained weak externally and was slightly lower for EG-VAN. HAM is seven-class and PH² has mapped NV/MEL truth only, so these are descriptive cross-dataset changes, not causal domain-shift magnitudes or directly equivalent population risks.","",
              "## Improvement diagnosis and claim boundaries", "",
              "The safest next model-development experiment is a pre-registered HAM train/validation-only melanoma-objective/sampler ablation with a fixed selection rule. This would be exploratory because the HAM test and PH² follow-up have already been viewed. A new untouched external dataset is needed for final confirmation. A new training run is scientifically justifiable only as such a pre-specified exploratory ablation, not as PH²-driven optimization. No new training was started.","",
              "`improvement_hypotheses.csv` separates HAM-only proposals, PH²-driven choices to reject, and untouched external confirmation. `claim_evidence_matrix.csv` gives safe paper wording. No claim of statistical or clinical superiority is supported here.",""]
    return "\n".join(lines)


if __name__=="__main__":
    main()
