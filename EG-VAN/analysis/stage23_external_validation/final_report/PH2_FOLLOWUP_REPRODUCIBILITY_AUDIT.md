# Stage 23 PH2 follow-up: reproducibility and integrity audit

**Disposition: PASS for integrity of the downloaded, completed follow-up.** This audit used the four saved outputs and frozen source files only. It did not load model weights into a network, preprocess images for inference, run training/inference, or change the checkpoint, split, mapping, threshold or original `run/` files.

## Artifact and source verification

| Artifact / source | SHA256 | Audit finding |
|---|---|---|
| `run/manifest.json` | `c8c570582bb06ff16955f2792fc8b65ba5885875a2328bd7052364f7ebc10e4b` | Status `COMPLETE_QUALIFIED_EXTERNAL_FOLLOWUP`; Tesla T4, torch 2.11.0+cu130, CUDA 13.0, Python 3.13.15, physical batch 16; flags state no training, threshold or checkpoint change. These flags are provenance records, not independent proof of the entire remote process. |
| `run/predictions.csv` | `2b131ac9ad0e388036194f463914d5e5a96f3bd1934470a38266051d93f341e4` | Matches manifest; 120 unique IDs in frozen manifest order, 80 NV / 40 MEL labels, seven valid probabilities and consistent argmax/mapping/entropy/confidence for every row. |
| `run/metrics.json` | `6aec707298708bb7e3570083489e00a4402b6e02abc823c80b980f333a435356` | Matches manifest; all saved point estimates independently reproduced from predictions, including confusion, sensitivity, specificity, mapped balanced accuracy, ROC AUC, AP and uncertainty summaries. Wilson and fixed-seed bootstrap intervals independently reproduced. |
| `run/confusion_matrix.csv` | `26db8800bab80466170d17d0e446ebd0961a7512413d23b968e97f5b290a7e75` | Matches manifest and independently recomputed matrix `[[68,5,7],[13,13,14]]`. |
| Frozen Stage 23 checkpoint | `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab` | Independently hashed, matching manifest and frozen model registry; epoch-14 identity is registry-backed. This audit did not deserialize it. |
| Frozen split | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` | Independently hashed; split preserved. HAM test outcomes were not read. |
| Frozen PH2 protocol | `3e6e31165e6afcb6b270e940f3abfe94dc99e2c805d84e08db2a156a91754ed8` | Independently hashed; fixed cohort, mapping, endpoints and numerical path match manifest. |
| PH2 cohort manifest | `3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842` | Independently hashed; 200 total, 120 included, 80 atypical excluded. All 200 source-image SHA256 values match manifest and local files. |
| PH2 clinical table | `2cc2fdc28eb230d86b56d72a5aa710593d0861cf6564fa64f43a97a89070ccf4` | Independently hashed. |
| Frozen inference runner | `7eee502b91992aae15b3c130809534ee062637da4ad47febeebf54f512025b0f` | Independently hashed, matching manifest. Source uses frozen loader, CUDA FP16 batch 16 and selective FP32 nonlocal3 q@k. |
| Preprocessing source | `cf0a176a610b52ba45b0e91761b027a560252234f81f7b6fd572f02d8baab6ed` | Independently hashed, matching manifest and registry. |

The seven-class order is `akiec,bcc,bkl,df,mel,nv,vasc`. Each probability is finite and in [0,1]; per-row sums range approximately **0.9999998811 to 1.0000001150**, comfortably within the frozen `1e-5` check. Every saved confidence equals the maximum class probability, every saved entropy equals `−Σp log p`, and every mapped/binary label matches the seven-class argmax. Prediction IDs and truth labels match the fixed 120-row included manifest in order. The output lacks patient IDs because the PH2 package does not provide a patient linkage field.

## Independent computation

Run `python analysis/stage23_external_validation/audit_saved_followup.py` from the repository root for a read-only repeat. [Audit source](../audit_saved_followup.py) checks source/output hashes, all 200 image hashes, 120 predictions, the frozen mapping, confusion CSV, all saved scalar metrics, Wilson bounds and the four stored fixed-seed bootstrap intervals. The completed [machine audit](recomputed_audit.json) has status `PASS`. [Metrics CSV](METRICS_TABLE.csv) is a derived publication table in this separate folder. The additional descriptive mapped balanced-accuracy percentile interval uses 2,000 fixed-seed class-stratified **image-ID** resamples; lack of patient IDs prevents a patient-cluster interval.

## What cannot be proved

The manifest and file hashes support the reported remote execution and current artifact identity; they do not independently prove every in-memory CUDA operation, byte identity to an unrecorded reference Stage 23 PH2 run, or patient independence. The Stage 23 training manifest records `ph2_accessed=false`, but earlier project-level PH2 diagnostics and older-model evaluations are documented. Thus the evidence is a **qualified external follow-up**, not an untouched independent validation. The 80 atypical nevi are excluded; reported estimates apply only to the restricted 120-case mapping. No clinical adequacy or deployment claim follows from this audit.
