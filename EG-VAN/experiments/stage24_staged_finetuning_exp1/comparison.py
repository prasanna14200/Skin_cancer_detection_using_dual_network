"""Deterministic Stage24 validation comparison against frozen Stage15/23 records."""
from __future__ import annotations

AGGREGATE_METRICS = (
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
    "macro_f1", "weighted_f1", "mel_precision", "mel_recall", "mel_f1", "nv_recall",
)
PRIMARY_METRICS = ("accuracy", "macro_f1", "mel_recall", "mel_f1")


def summarize(metrics: dict) -> dict:
    per_class = metrics["per_class"]
    support = sum(values["support"] for values in per_class.values())
    return {
        "epoch": metrics.get("epoch"),
        "validation_loss": metrics["validation_loss"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics.get("macro_precision",
                           sum(v["precision"] for v in per_class.values()) / len(per_class)),
        "macro_recall": metrics.get("macro_recall",
                        sum(v["recall"] for v in per_class.values()) / len(per_class)),
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics.get("weighted_f1", sum(v["f1"] * v["support"] for v in per_class.values()) / support),
        "mel_precision": per_class["mel"]["precision"],
        "mel_recall": per_class["mel"]["recall"],
        "mel_f1": per_class["mel"]["f1"],
        "nv_recall": per_class["nv"]["recall"],
        "mel_tp": per_class["mel"]["tp"],
        "mel_fn": per_class["mel"]["fn"],
    }


def compare_stage24(stage24: dict | None, stage23: dict, stage15: dict) -> dict:
    summary15, summary23 = summarize(stage15), summarize(stage23)
    if stage24 is None:
        return {
            "decision": "NO_CANDIDATE_SELECTED",
            "stage15": summary15,
            "stage23": summary23,
            "stage24": None,
            "classwise": None,
            "confusion_matrices": {
                "stage15": stage15["confusion_matrix"],
                "stage23": stage23["confusion_matrix"],
                "stage24": None,
            },
        }
    if not stage24.get("eligible"):
        raise ValueError("Stage24 comparison requires the selected eligible validation record")
    if not (stage15["class_order"] == stage23["class_order"] == stage24["class_order"]):
        raise ValueError("Reference and Stage24 class orders differ")
    classwise = {}
    class_no_regression = True
    class_strict_improvement = False
    for name in stage24["class_order"]:
        before23, candidate = stage23["per_class"][name], stage24["per_class"][name]
        before15 = stage15["per_class"][name]
        if not (before15["support"] == before23["support"] == candidate["support"]):
            raise ValueError(f"Reference support mismatch for {name}")
        values = {}
        for metric in ("precision", "recall", "f1"):
            old23, new = before23[metric], candidate[metric]
            old15 = before15[metric]
            values[metric] = {
                "stage15": old15,
                "stage23": old23,
                "stage24": new,
                "delta_vs_stage15": new - old15,
                "delta_vs_stage23": new - old23,
            }
            class_no_regression &= new >= old23
            class_strict_improvement |= new > old23
        classwise[name] = {
            **values,
            "support": candidate["support"],
            "tp_fp_fn_stage15": {k: before15[k] for k in ("tp", "fp", "fn")},
            "tp_fp_fn_stage23": {k: before23[k] for k in ("tp", "fp", "fn")},
            "tp_fp_fn_stage24": {k: candidate[k] for k in ("tp", "fp", "fn")},
        }
    summary24 = summarize(stage24)
    aggregate_delta23 = {m: summary24[m] - summary23[m] for m in AGGREGATE_METRICS}
    aggregate_delta15 = {m: summary24[m] - summary15[m] for m in AGGREGATE_METRICS}
    aggregate_no_regression = all(value >= 0 for value in aggregate_delta23.values())
    loss_no_regression = summary24["validation_loss"] <= summary23["validation_loss"]
    mel_gain_preserved = (summary24["mel_recall"] >= summary23["mel_recall"] and
                          summary24["mel_f1"] >= summary23["mel_f1"])
    strict = (any(value > 0 for value in aggregate_delta23.values()) or
              class_strict_improvement or summary24["validation_loss"] < summary23["validation_loss"])
    primary_improved = any(aggregate_delta23[m] > 0 for m in PRIMARY_METRICS)
    clear = (aggregate_no_regression and class_no_regression and loss_no_regression and
             mel_gain_preserved and strict)
    if clear:
        decision = "CLEAR_VALIDATION_IMPROVEMENT"
    elif primary_improved:
        decision = "MIXED_VALIDATION_RESULT"
    else:
        decision = "NO_MEANINGFUL_IMPROVEMENT"
    order = stage24["class_order"]
    mel_idx, nv_idx = order.index("mel"), order.index("nv")
    return {
        "decision": decision,
        "stage15": summary15,
        "stage23": summary23,
        "stage24": summary24,
        "aggregate_delta_vs_stage15": aggregate_delta15,
        "aggregate_delta_vs_stage23": aggregate_delta23,
        "classwise": classwise,
        "mel_tp_fn": {
            "stage15": {"tp": summary15["mel_tp"], "fn": summary15["mel_fn"]},
            "stage23": {"tp": summary23["mel_tp"], "fn": summary23["mel_fn"]},
            "stage24": {"tp": summary24["mel_tp"], "fn": summary24["mel_fn"]},
        },
        "mel_to_nv": {
            "stage15": stage15["confusion_matrix"][mel_idx][nv_idx],
            "stage23": stage23["confusion_matrix"][mel_idx][nv_idx],
            "stage24": stage24["confusion_matrix"][mel_idx][nv_idx],
        },
        "nv_to_mel": {
            "stage15": stage15["confusion_matrix"][nv_idx][mel_idx],
            "stage23": stage23["confusion_matrix"][nv_idx][mel_idx],
            "stage24": stage24["confusion_matrix"][nv_idx][mel_idx],
        },
        "confusion_matrices": {
            "stage15": stage15["confusion_matrix"],
            "stage23": stage23["confusion_matrix"],
            "stage24": stage24["confusion_matrix"],
        },
        "decision_checks": {
            "no_aggregate_regressions_vs_stage23": aggregate_no_regression,
            "no_classwise_regressions_vs_stage23": class_no_regression,
            "validation_loss_no_worse_than_stage23": loss_no_regression,
            "stage23_mel_recall_and_f1_preserved": mel_gain_preserved,
        },
    }
