"""Analyze complete CUDA Phase B saved predictions; never performs inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_cuda_full_validation import (HERE, ROOT, FROZEN, CLASSES, CHECKPOINT_SHA,
    batches, existing_batches, frozen_inputs, sha, complete_baseline_gate)

PREDICTIONS = HERE / "cuda_full_validation_predictions.csv"


def paired_statistics(df, condition, baseline):
    paired = condition.merge(baseline[["image_id","predicted_class","correct","entropy_nats"]],
                             on="image_id", suffixes=("","_baseline"), validate="one_to_one")
    if len(paired)!=986 or paired.image_id.duplicated().any():
        raise ValueError("Paired condition is incomplete or duplicate")
    y = paired.true_class.to_numpy()
    pred = paired.predicted_class.to_numpy()
    base_pred = paired.predicted_class_baseline.to_numpy()
    mel = y == "mel"
    return {
        "images": len(paired), "lesions": paired.lesion_id.nunique(),
        "accuracy": float((pred==y).mean()),
        "macro_f1": float(f1_score(y,pred,labels=CLASSES,average="macro",zero_division=0)),
        "mel_recall": float((pred[mel]=="mel").mean()), "mel_support": int(mel.sum()),
        "mel_correct": int(((pred=="mel") & mel).sum()),
        "mel_to_nv": int(((pred=="nv") & mel).sum()),
        "prediction_flips": int((pred!=base_pred).sum()),
        "prediction_flip_rate": float((pred!=base_pred).mean()),
        "correct_to_incorrect": int(((base_pred==y)&(pred!=y)).sum()),
        "incorrect_to_correct": int(((base_pred!=y)&(pred==y)).sum()),
        "mean_entropy_nats": float(paired.entropy_nats.mean()),
        "median_entropy_nats": float(paired.entropy_nats.median()),
        "mean_entropy_delta": float((paired.entropy_nats-paired.entropy_nats_baseline).mean()),
        "median_entropy_delta": float((paired.entropy_nats-paired.entropy_nats_baseline).median()),
        "accuracy_delta": float(((pred==y).astype(int)-(base_pred==y).astype(int)).mean()),
        "macro_f1_delta": float(f1_score(y,pred,labels=CLASSES,average="macro",zero_division=0)-
                                f1_score(y,base_pred,labels=CLASSES,average="macro",zero_division=0)),
        "mel_recall_delta": float((pred[mel]=="mel").mean()-(base_pred[mel]=="mel").mean()),
    }, paired


def bootstrap(paired, seed, repetitions=1000):
    groups = paired.groupby("lesion_id").indices
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    out = {key: [] for key in ("accuracy_delta","macro_f1_delta","mel_recall_delta","prediction_flip_rate","mean_entropy_delta")}
    for _ in range(repetitions):
        idx = np.concatenate([groups[k] for k in rng.choice(keys,len(keys),replace=True)])
        sample = paired.iloc[idx]
        y = sample.true_class.to_numpy()
        p = sample.predicted_class.to_numpy()
        b = sample.predicted_class_baseline.to_numpy()
        mel = y=="mel"
        out["accuracy_delta"].append(float(((p==y).astype(int)-(b==y).astype(int)).mean()))
        out["macro_f1_delta"].append(float(f1_score(y,p,labels=CLASSES,average="macro",zero_division=0)-
                                           f1_score(y,b,labels=CLASSES,average="macro",zero_division=0)))
        if mel.any():
            out["mel_recall_delta"].append(float((p[mel]=="mel").mean()-(b[mel]=="mel").mean()))
        out["prediction_flip_rate"].append(float((p!=b).mean()))
        out["mean_entropy_delta"].append(float((sample.entropy_nats-sample.entropy_nats_baseline).mean()))
    return {key: {"percentile_95": [float(x) for x in np.percentile(vals,[2.5,97.5])],
                  "valid_repetitions": len(vals)} for key,vals in out.items()}


def make_plots(summary):
    x = np.arange(len(summary))
    names = summary.condition.tolist()
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for key,label in (("accuracy","Accuracy"),("macro_f1","Macro F1"),("mel_recall","MEL recall")):
        axes[0].plot(x,summary[key],marker="o",label=label)
    axes[0].set(ylabel="Score",ylim=(0,1),title="Frozen Stage 23 full-validation synthetic robustness")
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(x,summary.mean_entropy_delta,marker="o")
    axes[1].axhline(0,color="gray",ls="--")
    axes[1].set(ylabel="Paired mean entropy change (nats)")
    axes[1].grid(alpha=.2)
    for ax in axes: ax.set_xticks(x,names,rotation=45,ha="right")
    fig.tight_layout();fig.savefig(HERE/"cuda_robustness_curves.png",dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(x-.2,summary.correct_to_incorrect,width=.4,label="Correct → incorrect")
    ax.bar(x+.2,summary.incorrect_to_correct,width=.4,label="Incorrect → correct")
    ax.set(xticks=x,xticklabels=names,ylabel="Validation images",title="Paired classification transitions")
    ax.tick_params(axis="x",rotation=45);ax.legend();fig.tight_layout()
    fig.savefig(HERE/"cuda_prediction_transitions.png",dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(x,summary.mel_correct,marker="o",label="MEL correctly identified")
    ax.plot(x,summary.mel_to_nv,marker="s",label="MEL → NV")
    ax.set(xticks=x,xticklabels=names,ylabel="Cases of 107",title="Validation melanoma outcomes")
    ax.tick_params(axis="x",rotation=45);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(HERE/"cuda_melanoma_robustness.png",dpi=300);plt.close(fig)


def main():
    if not PREDICTIONS.is_file():
        raise FileNotFoundError("Complete 6,902-row inference table is required; no partial analysis")
    protocol, validation, reference, input_hashes = frozen_inputs()
    protocol_sha = sha(FROZEN)
    gate = complete_baseline_gate(validation,reference,protocol_sha)
    table = pd.read_csv(PREDICTIONS)
    names = [x["name"] for x in protocol["degradation_conditions"]]
    if len(table)!=6902 or table.duplicated(["image_id","condition"]).any():
        raise ValueError("Final table incomplete or duplicated")
    for name in names:
        committed = existing_batches(name,validation,reference,protocol_sha)
        if len(committed)!=len(batches(validation)):
            raise ValueError(f"Incomplete committed batch set: {name}")
        expected = pd.concat([committed[i] for i in range(len(batches(validation)))],ignore_index=True)
        actual = table.loc[table.condition.eq(name)].reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"Final table differs from committed batches: {name}")
    baseline = table.loc[table.condition.eq("baseline")]
    summaries, intervals, by_class = [], {}, {}
    for i,name in enumerate(names):
        sub = table.loc[table.condition.eq(name)]
        stats, paired = paired_statistics(table,sub,baseline)
        summaries.append({"condition":name,**stats})
        intervals[name] = bootstrap(paired,seed=2310+i)
        by_class[name] = {cls: {"support":int(len(q)),"correct":int(q.correct.sum()),
                                "recall":float(q.correct.mean())} for cls,q in sub.groupby("true_class")}
    summary=pd.DataFrame(summaries)
    summary.to_csv(HERE/"cuda_summary.csv",index=False,float_format="%.17g")
    table.loc[table.true_class.eq("mel")].to_csv(HERE/"cuda_melanoma_analysis.csv",index=False,float_format="%.17g")
    statistical={"status":"COMPLETE_FULL_VALIDATION","scope":"986 Stage 23 validation images only",
                 "baseline_gate":gate,"conditions":summaries,"classwise":by_class,
                 "paired_lesion_cluster_bootstrap_95":intervals,"bootstrap_seed":2310,"bootstrap_repetitions":1000,
                 "multiple_comparisons":"exploratory; no multiplicity correction or degradation tuning",
                 "clinical_quality_validation":False,"training_performed":False,"test_or_ph2_accessed":False}
    (HERE/"cuda_statistical_results.json").write_text(json.dumps(statistical,indent=2,allow_nan=False)+"\n")
    make_plots(summary)
    output_names=["cuda_full_validation_predictions.csv","baseline_gate.json","cuda_summary.csv",
                  "cuda_melanoma_analysis.csv","cuda_statistical_results.json","cuda_robustness_curves.png",
                  "cuda_prediction_transitions.png","cuda_melanoma_robustness.png"]
    manifest={"status":"COMPLETE_FULL_VALIDATION","checkpoint_sha256":CHECKPOINT_SHA,
              "frozen_input_sha256":input_hashes,"protocol_sha256":protocol_sha,
              "runner_sha256":sha(HERE/"run_cuda_full_validation.py"),"analysis_sha256":sha(Path(__file__)),
              "gpu":torch_device_from_batches(validation,reference,protocol_sha),
              "batch_size":16,"autocast_dtype":"torch.float16","images":986,"predictions":6902,
              "output_sha256":{name:sha(HERE/name) for name in output_names},
              "training_performed":False,"test_or_ph2_accessed":False}
    report=make_report(summary,gate,manifest,intervals)
    (HERE/"FINAL_CUDA_PHASE_B_REPORT.md").write_text(report,encoding="utf-8")
    manifest["output_sha256"]["FINAL_CUDA_PHASE_B_REPORT.md"]=sha(HERE/"FINAL_CUDA_PHASE_B_REPORT.md")
    (HERE/"cuda_execution_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"COMPLETE_FULL_VALIDATION","baseline_max_delta":gate["max_abs_probability_delta"],
                      "gpu":manifest["gpu"],"predictions":6902},indent=2))


def torch_device_from_batches(validation,reference,protocol_sha):
    metadata=[]
    for name in ("baseline","jpeg_q40"):
        for i in (0,len(batches(validation))-1):
            path=HERE/"batches"/name/f"batch_{i:04d}"/"metadata.json"
            metadata.append(json.loads(path.read_text()))
    gpu={m["gpu"] for m in metadata}
    versions={m["torch_version"] for m in metadata}
    if len(gpu)!=1 or len(versions)!=1:
        raise ValueError("Mixed CUDA runtime in committed batches")
    return {"device":gpu.pop(),"torch_version":versions.pop()}


def make_report(summary,gate,manifest,intervals):
    lines=["# Stage 23 Phase B full-validation CUDA robustness report", "",
           "The frozen Stage 23 checkpoint was evaluated on **all 986 validation images** in the original split order with batch size 16 and CUDA FP16 autocast. No training or checkpoint selection occurred. HAM test and PH2 were not accessed.","",
           f"Checkpoint SHA256: `{CHECKPOINT_SHA}`. GPU/runtime: `{manifest['gpu']['device']}`, PyTorch `{manifest['gpu']['torch_version']}`. Selective FP32 nonlocal3 q@k remains installed by the frozen loader.","",
           f"Complete baseline gate: **PASS**, 986/986 labels agree with saved Stage 23 predictions; maximum seven-class probability delta `{gate['max_abs_probability_delta']:.9g}` (required ≤0.005). No degraded condition began until this gate passed.","",
           "| Condition | Accuracy | Macro F1 | MEL recall | Flips | Mean entropy Δ (nats) |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.condition} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.mel_recall:.4f} ({row.mel_correct}/107) | {row.prediction_flips}/986 | {row.mean_entropy_delta:+.4f} |")
    lines += ["","All changes are paired with the unchanged baseline. `cuda_statistical_results.json` records classwise support and recall, correct→incorrect/incorrect→correct transitions, and 1,000-repetition lesion-cluster percentile intervals (seed 2310) for accuracy, macro F1, melanoma recall, flip rate, and entropy changes.","",
              "These fixed synthetic raw-image perturbations are exploratory. They do not measure clinical technical adequacy, establish a quality gate, or justify threshold tuning. PH2 and HAM test outcomes were outside this validation-only study.","",
              "The outputs are reproducible from `run_cuda_full_validation.py` and `analyze_cuda_full_validation.py`; `cuda_execution_manifest.json` contains frozen input, source and output SHA256 hashes. Existing CPU chunks in the separate `phase_b_full_validation/` directory were not combined or modified.",""]
    return "\n".join(lines)


if __name__=="__main__":
    main()
