"""Validation-only comparison against the frozen Stage15 epoch-16 metrics."""
from __future__ import annotations


HIGHER_AGGREGATES = (
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
    "macro_f1", "weighted_f1", "mel_recall", "mel_f1", "nv_recall",
)
PRIMARY_IMPROVEMENTS = ("accuracy", "macro_f1", "mel_recall", "mel_f1")


def _summary(metrics: dict) -> dict:
    per_class = metrics["per_class"]
    support_total = sum(values["support"] for values in per_class.values())
    return {
        "epoch": metrics["epoch"],
        "validation_loss": metrics["validation_loss"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": sum(values["precision"] for values in per_class.values()) / len(per_class),
        "macro_recall": sum(values["recall"] for values in per_class.values()) / len(per_class),
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": sum(values["f1"] * values["support"] for values in per_class.values()) / support_total,
        "mel_recall": per_class["mel"]["recall"],
        "mel_f1": per_class["mel"]["f1"],
        "nv_recall": per_class["nv"]["recall"],
    }


def compare_validation(stage23: dict | None, stage15: dict) -> dict:
    reference = _summary(stage15)
    if stage23 is None:
        return {
            "decision": "NO_CANDIDATE_SELECTED",
            "stage15_validation": reference,
            "stage23_validation": None,
            "classwise": None,
            "confusion_matrices": {"stage15": stage15["confusion_matrix"], "stage23": None},
        }
    if not stage23.get("eligible"):
        raise ValueError("Stage23 comparison requires a checkpoint selected by the frozen gates")
    candidate = _summary(stage23)
    if stage15["class_order"] != stage23["class_order"]:
        raise ValueError("Stage15 and Stage23 validation class orders differ")
    classes = stage15["per_class"].keys()
    if set(classes) != set(stage23["per_class"]):
        raise ValueError("Stage15 and Stage23 validation classes differ")
    classwise = {}
    per_class_no_worse = True
    per_class_strict = False
    for name in stage15["class_order"]:
        before, after = stage15["per_class"][name], stage23["per_class"][name]
        if before["support"] != after["support"]:
            raise ValueError(f"Validation support changed for {name}")
        values = {}
        for metric in ("precision", "recall", "f1"):
            old, new = before[metric], after[metric]
            values[metric] = {"stage15": old, "stage23": new, "delta": new - old}
            per_class_no_worse &= new >= old
            per_class_strict |= new > old
        classwise[name] = {
            **values,
            "support": before["support"],
            "stage15_tp_fp_fn": {key: before[key] for key in ("tp", "fp", "fn")},
            "stage23_tp_fp_fn": {key: after[key] for key in ("tp", "fp", "fn")},
        }
    delta = {key: candidate[key] - reference[key] for key in HIGHER_AGGREGATES}
    aggregate_no_worse = all(delta[key] >= 0 for key in HIGHER_AGGREGATES)
    lower_or_equal_loss = candidate["validation_loss"] <= reference["validation_loss"]
    strict_improvement = (any(value > 0 for value in delta.values()) or per_class_strict or
                          candidate["validation_loss"] < reference["validation_loss"])
    no_regressions = aggregate_no_worse and per_class_no_worse and lower_or_equal_loss
    primary_improvement = any(delta[key] > 0 for key in PRIMARY_IMPROVEMENTS)
    if no_regressions and strict_improvement:
        decision = "CLEAR_VALIDATION_IMPROVEMENT"
    elif primary_improvement:
        decision = "MIXED_VALIDATION_RESULT"
    else:
        decision = "NO_MEANINGFUL_IMPROVEMENT"
    mel_index = stage23["class_order"].index("mel")
    nv_index = stage23["class_order"].index("nv")
    return {
        "decision": decision,
        "stage15_validation": reference,
        "stage23_validation": candidate,
        "aggregate_delta_stage23_minus_stage15": delta,
        "classwise": classwise,
        "mel_tp_fn": {
            "stage15": {"tp": stage15["per_class"]["mel"]["tp"], "fn": stage15["per_class"]["mel"]["fn"]},
            "stage23": {"tp": stage23["per_class"]["mel"]["tp"], "fn": stage23["per_class"]["mel"]["fn"]},
        },
        "mel_to_nv": {
            "stage15": stage15["confusion_matrix"][mel_index][nv_index],
            "stage23": stage23["confusion_matrix"][mel_index][nv_index],
        },
        "nv_to_mel": {
            "stage15": stage15["confusion_matrix"][nv_index][mel_index],
            "stage23": stage23["confusion_matrix"][nv_index][mel_index],
        },
        "confusion_matrices": {
            "stage15": stage15["confusion_matrix"],
            "stage23": stage23["confusion_matrix"],
        },
        "no_regressions_on_registered_metrics": no_regressions,
    }
