# Phase 8 - PH² Provenance and Access Audit

Date: 2026-09-26
Status: **BLOCKED — DATA PROVENANCE INSUFFICIENT** (evaluator preparation and dry-run audit passed; inference not run)

This is an audit only. No download, conversion, inference, training, or PH² file modification was performed.

## 1. Objective

Determine whether the PH² material currently available can be used legitimately and reproducibly for external validation of the frozen leakage-aware HAM10000 EfficientNetV2S checkpoint.

## 2. Official PH² source and citation

- Official database page: University of Porto, Faculty of Sciences, ADDI PH² Database: https://www.fc.up.pt/addi/ph2%20database.html
- The page identifies the PH² Database and has sections for data description, terms of use, download, and support. It also exposes a restricted/login area.
- The public page fetch exposes the section names and login link, but not the detailed terms/download content. This audit cannot verify a direct public download URL, whether a particular account/approval is required, or the exact license/redistribution conditions.
- PH² publication citation verified from the Crossref DOI record: Mendonça, T., Ferreira, P. M., Marques, J. S., Marçal, A. R. S., & Rozeira, J. (2013). “PH² - A dermoscopic image database for research and benchmarking.” 2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 5437–5440. DOI: `10.1109/EMBC.2013.6610779`.

## 3. Official access and provenance status

**Official access: unresolved.** Earlier project documentation records that the official University download workflow did not yield a package after waiting for access. No official University archive or access approval is present locally.

**Secondary source:** the local package directory is named `data/external/PH2Dataset/` and the project documentation identifies it as a Kaggle-hosted secondary mirror. However, the Kaggle URL, exact Kaggle dataset name/uploader, revision/version, acquisition date, and license/terms file are absent from the synchronized project. There is no original Kaggle archive file to hash; the delivered content is already extracted.

Only this one unidentified secondary package is present locally. The local tree does not establish which Kaggle revision it came from or whether the uploader's copy is identical to the official release; no claim about the number of versions/mirrors available elsewhere is made.

The files are internally consistent with the expected PH² structure, but internal consistency does not establish that this copy is identical to the official distribution or that its reuse terms permit this study.

**Provenance classification: PARTIALLY VERIFIED.** Dataset identity/label structure is supported by the bundled PH²-named files and expected structure; chain of custody, source revision, authenticity, and license/access permission are not independently verified. The copy is **not** treated as an official University download.

## 4. Exact local PH² structure

The supplied extracted mirror is at `data/external/PH2Dataset/`. A preserved byte-for-byte copy is at `data/external/ph2/original/`.

| Location | Contents |
|---|---|
| `data/external/ph2/original/` | 453 files: `PH2_dataset.txt`, `PH2_dataset.xlsx`, `Readme.txt`, and 450 BMPs in 200 case folders |
| `data/external/ph2/metadata/` | PH² text metadata, workbook, and derived `ph2_manifest.csv` |
| `data/external/ph2/images/` | 200 copied original dermoscopic BMPs plus a pre-existing `.gitkeep` |
| Original package BMP breakdown | 200 dermoscopic images, 200 lesion masks, 50 ROI/color masks |

The 200 files in `images/` intentionally duplicate the 200 original dermoscopic images contained in `original/`; they are separate working copies, not additional PH² cases. The preserved mirror tree and `original/` copy each contain 453 files with matching relative paths and SHA256 for every file (zero missing, extra, or hash-mismatched files).

- Image format: BMP.
- Original image IDs: 200 unique `IMD###` IDs.
- Dimensions: variable, approximately 553–577 pixels high by 761–769 pixels wide in the observed package.
- Readability: 200/200 original dermoscopic images opened with OpenCV.
- Duplicate image content among the 200 originals: zero duplicate-content groups.
- License/terms/citation-specific files in the package: none. Bundled `Readme.txt` explains folder contents but does not establish source URL or usage rights.
- Suspicious/incomplete provenance indicators: no archive, URL, uploader/revision record, license, or permission record. The `.gitkeep` is a placeholder in the derived images directory, not a PH² image.

## 5. Metadata and label audit

`PH2_dataset.txt` and the workbook contain 200 case IDs and a clinical diagnosis field. The text legend states:

- `0` = Common Nevus
- `1` = Atypical Nevus
- `2` = Melanoma

Verified counts from the bundled clinical diagnosis field:

| Clinical diagnosis | Code | Count |
|---|---:|---:|
| Common nevus | 0 | 80 |
| Atypical nevus | 1 | 80 |
| Melanoma | 2 | 40 |

All 200 clinical diagnosis values are present, IDs are unique, and all 200 labels match one original dermoscopic image ID. No label-only IDs or image-only IDs were found. Histological diagnosis is a separate field and is blank for many cases; the conservative evaluation mapping relies only on the explicit clinical diagnosis field and must not substitute histology or infer missing diagnoses.

The available metadata uses one `IMD###` case/image identifier per record. No separate patient identifier or patient-to-lesion linkage table was found in the package. Thus the files support image-to-clinical-diagnosis mapping, not patient-level grouping claims.

## 6. Intended mapping and exclusions

| PH² clinical diagnosis | HAM10000 label | Audit decision |
|---|---|---|
| Common nevus | `nv` | **YES**, direct mapping supported by the bundled clinical label code |
| Melanoma | `mel` | **YES**, direct mapping supported by the bundled clinical label code |
| Atypical nevus | None | **EXCLUDED**; never map to `bkl` or another HAM10000 class |

If provenance/access is later approved, the label-based candidate overlap is 120 images (80 common nevus + 40 melanoma); 80 atypical-nevus images are excluded. This is an eligibility count from the secondary mirror, not an inference result.

## 7. Checkpoint audit

- Path: `experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt`
- Exists: **YES**
- Size: 81,652,547 bytes
- SHA256: `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`
- PyTorch checkpoint deserialization: **PASS**
- Recorded model/config: EfficientNetV2S, leakage-aware split, seven frozen classes, 384x384, ImageNet pretrained weights.
- Classifier state shape: weight `(7, 1280)`, bias `(7,)`.
- Inference execution: **NOT RUN**.

Static state/config checks are consistent with the baseline definition. A live torchvision model construction was not used to execute inference in this audit.

## 8. External evaluator readiness

The evaluator issues were corrected in `src/external_eval.py`:

1. **Weights/transform bug:** evaluation no longer tries `weights.transforms()` when `pretrained=False` returns `weights=None`. The evaluation transform is explicit and matches the trained deterministic test transform: resize to `(384,384)`, tensor conversion, ImageNet mean/std normalization; no augmentation or Phase 4B preprocessing rerun.
2. **Atypical-nevus handling:** all 200 diagnosis rows are audited first. Common nevus maps to `nv`, melanoma to `mel`, and atypical nevus is explicitly recorded as excluded (80); unknown labels fail closed.
3. **Out-of-overlap predictions:** the seven-way argmax is preserved. The confusion matrix has true rows `(nv,mel)` and predicted columns `(nv,mel,other)`. All predictions into any of the other five HAM10000 classes remain visible under `other` and count as errors for binary accuracy/precision/recall/specificity/F1; no predictions are dropped.

The binary convention treats melanoma as positive: TP is true melanoma predicted mel; FN includes true melanoma predicted nv or other; TN is true nevus predicted nv; FP includes true nevus predicted mel or other.

## 9. Dry-run and synthetic tests

- Python compilation: PASS.
- Synthetic unit tests: 11/11 PASS, covering common-nevus and melanoma mapping, atypical exclusion, unknown labels, metadata mismatch, duplicate IDs, missing images, 2x3 confusion matrix, out-of-overlap predictions, and deterministic 384x384 transform creation.
- Real-package structural dry-run: PASS. Verified 200 total cases; counts 80/80/40; 120 mapped and 80 atypical excluded; 200 source dermoscopic images, 200 lesion masks, and 50 ROI masks; image IDs align to clinical labels.
- Checkpoint: PASS. Strict state loading into the repository EfficientNetV2S architecture succeeded; classifier shape `(7,1280)` plus `(7,)` bias.
- Runtime: CUDA unavailable locally (`torch 2.11.0+cpu`), so dry-run reports `inference_ready: false`. No forward pass or external inference ran.
- Provenance gate: PARTIALLY VERIFIED, so normal inference is refused even on a GPU until a verified provenance record is provided.

The evaluator requires `data/external/ph2/metadata/provenance.json` with status `VERIFIED` and non-empty dataset URL/name, uploader, revision, download date, license/terms, and traceability evidence before normal inference can proceed.

## 10. Readiness decision

**C. BLOCKED — DATA PROVENANCE INSUFFICIENT**

The image-label structure is complete enough to apply the intended direct mapping, but the available package does not establish the Kaggle source/revision, chain of custody to the official PH² release, or license/access permission. Therefore this project cannot yet describe evaluation on the current copy as legitimate, independently traceable PH² external validation. No PH² inference is authorized or reported here.

## 11. Fastest legitimate next action

Provide the exact Kaggle dataset URL/name, uploader, version/revision, acquisition date, and the dataset page's license/provenance text; then check whether those records trace the files to the cited PH² release and permit the intended research use. If that cannot be established, request/obtain the package and terms directly through the official ADDI PH² access workflow. Do not call the Kaggle mirror official.

After provenance/access is resolved and the provenance record is verified, run the prepared evaluator in CUDA Colab. Do not train or fine-tune on PH².

## 12. Future CUDA command (not run)

```bash
%cd /content/drive/MyDrive/EG-VAN
!PYTHONPATH=src python -u src/external_eval.py \
	--project-root /content/drive/MyDrive/EG-VAN \
	--checkpoint /content/drive/MyDrive/EG-VAN/experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt \
	--output-dir /content/drive/MyDrive/EG-VAN/experiments/ph2_external_eval
```

This command will stop before inference unless provenance is verified and CUDA is available. No PH² metrics exist.

## 13. Path A Provenance Decision Addendum

Date: 2026-09-26
Decision: **VERIFIED for non-commercial academic research and external validation only**. This scoped decision supersedes the earlier evaluator-gate status above; the earlier audit findings remain as the history of why the original gate blocked.

### Evidence recorded separately

- **Original PH² source:** the official University of Porto ADDI PH² page identifies the dataset and says it may be used for research and educational purposes. It also says redistribution and commercial use are not allowed and publications using the dataset must cite the specified PH² paper. The same page says download follows a registration form.
- **Browser documentation:** the task-provided quotation states that the PH² database and Browser are only available for research and educational purposes. The official page links `PH2BrowserTutorial.pdf`; automated PDF text extraction was unavailable in this environment, so the quote is retained as supplied evidence rather than represented as independently extracted here.
- **Kaggle secondary listing:** Kaggle metadata identifies `spacesurfer/ph2-dataset`, dataset ID `6220095`, uploader Dmitrii K, version 2 dated 2024-12-03. Its license field remains exactly `Other (specified in description)` with no license URL. This field is not treated as a license or as the basis for research permission.
- **Source traceability:** VERIFIED for PH² source identity and local package/manifest concordance using the original metadata, matching image IDs and class counts, local file-integrity audit, Kaggle listing identity, and the linked report hash. The exact Kaggle archive bytes and local acquisition date are not retained, so byte-level linkage to version 2 remains unresolved.

### Gate change and limits

The former evaluator required a non-empty download date and generic `license_or_terms` value. That schema could not represent authoritative PH²-specific research/educational terms separately from Kaggle's `Other` metadata. The gate now validates the official PH² terms and restrictions, Kaggle identity and license-field literal, source-traceability evidence and report SHA256, citation requirement, and the narrow intended-use scope.

No explicit license grant is claimed. No commercial-use permission or redistribution permission is claimed. The new `VERIFIED` decision applies only to non-commercial academic research/external validation subject to the PH² citation requirement; it is not an unrestricted license or legal opinion.

### Validation and current readiness

- Synthetic evaluator tests: 16/16 PASS, including positive scoped-use evidence and negative missing-evidence/commercial/redistribution cases.
- Local structural dry-run: PASS; provenance `VERIFIED`; checkpoint architecture audit PASS; no forward pass or PH² inference.
- Runtime used for this dry-run: PyTorch 2.11.0+cpu, CUDA unavailable; `inference_ready: false`.
- Frozen HAM10000 data, splits, and checkpoint: not modified.
