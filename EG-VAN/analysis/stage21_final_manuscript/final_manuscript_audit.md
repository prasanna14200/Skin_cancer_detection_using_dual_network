# Stage 21 final manuscript integration audit

**Disposition: manuscript integration PASS; submission NOT READY.** This was documentation and formatting work only. No model training, scientific experiment, HAM/PH² classification inference, checkpoint/threshold change, test tuning or submission occurred.

## Backup and edited sources

The master and PLOS preparation source were backed up **before editing**:

| File | SHA256 |
|---|---|
| `egvan_reconstruction_final.pre_stage21.md` | `41f6bfb2c663ab2b5a793f3e8ccd9f04d60ce9f98d323deacdbe6b3d0fb65cad` |
| `plos_manuscript_source.pre_stage21.md` | `0e27006a360f940ce41f9fe6900ebace43129400cc3ab3526494d0bb767b0413` |
| Updated `manuscript/egvan_reconstruction_final.md` | `fc9eb7637f2ce4d4176c676d81f4d738690a879f2f2b5f2e55a6b12e876ce4fe` |
| Updated `manuscript/submission/plos_one/manuscript_source.md` | `2738a12741d7340d0881e38a0091001fcca7caccb2af4fd37707c12500586606` |

The PLOS source retains author/affiliation/corresponding-author placeholders, AI-assistance disclosure and acknowledgments placeholders, and supporting-information captions. Its reference list matches the updated master text. The draft cover letter and conversion instructions were brought into the new scope; they remain drafts requiring author review. The separate [submission freeze manifest](submission_freeze_manifest.json) records SHA256 values for 17 author-review files and explicitly records `submission_performed=false`.

## Numerical and scientific consistency

The read-only [audit script](audit_manuscript.py) compared the manuscript displays with frozen Stage 16 and Stage 20 JSON, checked Stage 16 output hashes against both manifests, and passed. Rounded table/text values are consistent with their full-precision sources; no new scientific result was calculated for the manuscript. The master and PLOS source both retain:

- HAM **full-cohort** n=1,014, accuracy `0.8234714003944773` (displayed 0.823471 / 82.3471%), macro F1 `0.699581633840868` (0.699582), macro OVR AUC `0.9582061035537619` (0.958206), MEL recall 56/107 (52.34%), MEL F1 0.568528 and NV recall 0.940828.
- PH² **included cohort** n=120: 80 NV, 40 MEL; 80 atypical nevi excluded. Accuracy 0.666667 (66.67%), MEL recall 11/40 (27.5%), MEL F1 0.392857, MEL probability AUC 0.542813 and NV recall 0.8625. PH² is consistently described as **external follow-up** with prior project use, not untouched validation.
- Validation-only entropy selection, threshold `0.7675495327940953`, approximately 80% target validation coverage. HAM retained 805/1,014 (79.39%), reviewed 209/1,014 (20.61%), retained-case accuracy 90.93%, captured 106/179 errors (59.22%) and 23/51 MEL false negatives. The 90.93% number is explicitly conditional on retained cases, never a replacement for 82.3471% full-cohort accuracy.
- Entropy error-detection AUROC validation/HAM/PH² 0.827386/0.858883/0.766875; ECE 0.019574/0.029678/0.060178; seven-class Brier 0.270595/0.256018/0.491627. PH² calibration is limited to the mapped included cohort.
- Validation quality-proxy analysis and its **non-monotonic** sharpness terciles (low 22.19% error, middle 14.33%); **no** quality labels, validated threshold or operational pre-model quality gate. Eight final-model `resnet.nonlocal3` Grad-CAM cases are qualitative; seven maps showed broad high response around the conspicuous central area, one BCC map more focal; no lesion masks, whole-model causal claim or clinical localization validation.
- FP16 q@k affinity overflow localized to `resnet.nonlocal3 torch.bmm(q,k)` and corrected by FP32 affinity under surrounding AMP. This is a numerical-stability correction, **not** an accuracy improvement. The original paper's approximately 98.2% nine-class result is identified as paper-reported and **not** a matched replication comparison.
- Recovered checkpoint SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`; exact observable replay but no byte-identity proof to the missing original epoch-16 weight file. Earlier HAM/PH² project exposure and exploratory timing of Stage 20 remain in Limitations. No clinical-readiness claim is present.

## Figure, table, citation and reference checks

Tables **1–6** and main figures **1–8** appear once, in numerical order, in each source. The master has eight existing relative figure links and all resolve. PLOS `Fig1.tif`–`Fig8.tif` exist: Fig1–4 are 300 dpi; Fig5–8 are 350 dpi. New TIFFs Fig5–8 have pixel content identical to their audited source images: HAM reliability diagram, HAM risk–coverage plot, four-case Grad-CAM panel and evidence-marked pipeline. Figure 8's caption makes clear that the quality-risk box before preprocessing depicts the **intended** order; actual proxies were computed offline on processed HAM images. The master abstract is 234 words by a simple whitespace/token count; author conversion should recheck.

In-text numeric citations [1]–[7] are all present; the seven numbered references exist in the same order in master and PLOS sources. This is an **internal** citation/reference audit; final bibliographic style, external metadata and publisher-format verification remain author tasks. Supporting figures S1–S4 and their captions remain in the PLOS preparation source.

## Dangerous/stale-claim search

Both manuscript sources were searched for `95%`, `98.2%`, `90.93%`, `external validation`, `quality gate`, `clinically validated`, `state-of-the-art`, `superior`, and `improved accuracy`. Every found `98.2%` identifies the original publication and explicitly disclaims direct comparison. Every `90.93%` occurrence identifies the selected retained subset or contrasts it with the full-cohort accuracy. `External validation` occurs only in a negated PH² limitation; `quality gate` occurs as absent/incomplete. Clinical validation is denied, and no unsupported state-of-the-art, superiority or accuracy-improvement claim was found. The PLOS source was checked after synchronization rather than assumed identical.

## Decision

The title and text may call the work **“EG-VAN+ as a reliability-extension prototype”**. The full original EG-VAN+ objective remains **PARTIAL** because a prospectively validated pre-classifier quality accept/reject gate was not achieved. This is an author-review manuscript, not a submission-ready package. Do not submit or upload automatically. See [claim-evidence matrix](claim_evidence_matrix.md) and [submission readiness](submission_readiness.md).
