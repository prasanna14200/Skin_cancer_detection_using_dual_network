# EG-VAN Stage B PH² External-Validation Plan

Date: 2026-09-23
Status: Historical preparation plan. Later secondary-mirror PH2 files were audited locally, but PH2 inference has not run.

## Official source verification

- Official source: University of Porto, Faculty of Sciences, ADDI project PH² Database page:
  `https://www.fc.up.pt/addi/ph2%20database.html`
- The institutional page identifies the PH² Database and exposes separate sections for Introduction, Data Description, Terms of Use, Download, and Benchmarking.
- Dataset identity reported by the official source: PH² dermoscopic image database.
- Commonly reported dataset composition to verify against the downloaded package: 200 dermoscopic images consisting of common nevi, atypical nevi, and melanoma. The exact package metadata and image count must be checked after authorized download rather than assumed.
- Image format: must be confirmed from the downloaded official package and metadata.
- Access/license: the official page provides Terms of Use and Download sections. No open permissive license was established from the accessible page; access conditions and any required academic-use attribution must be recorded from the official terms before use.

## Evaluation-only policy

- Do not retrain, fine-tune, calibrate, or tune the HAM10000 model on PH².
- Do not merge PH² into HAM10000.
- Use a completed HAM10000 baseline checkpoint only.
- Keep PH² artifacts under `data/external/ph2/` and never place them under HAM10000 raw or processed directories.

## Conservative label mapping

| PH² label | HAM10000 label | Decision | Reason |
|---|---|---|---|
| Common nevus | `nv` | Include if metadata confirms label | Direct clinical category overlap. |
| Melanoma | `mel` | Include if metadata confirms label | Direct clinical category overlap. |
| Atypical nevus | None | Exclude by default | Do not invent `bkl` or another biologically questionable mapping. |

The mapping must be confirmed against the official PH² metadata. If the package uses a different label vocabulary or lacks sufficient metadata, stop for approval rather than silently mapping.

## Planned verification checklist

1. Preserve the original downloaded archive and record source URL, access date, package name, and checksum.
2. Confirm image count, file extensions, metadata columns, labels, unreadable files, duplicates, and image/metadata correspondence.
3. Confirm the number of common-nevus and melanoma images eligible for evaluation.
4. Record all excluded atypical-nevus images and any other exclusions with exact reasons.
5. Apply only the already-frozen HAM10000 model input transform: resize to 384x384 and ImageNet normalization. Do not rerun hair removal, Gray World, or Retinex unless a separate approved external-data preprocessing decision is made.
6. Evaluate with accuracy, macro-F1, per-class recall, and confusion matrix only.
7. Do not add calibration or efficiency metrics yet.

## Planned project structure

```text
data/external/ph2/
  original/          # authorized source package, preserved
  metadata/          # source metadata and normalized evaluation manifest
  images/            # extracted PH² images only
src/external_eval.py
docs/PHASE_5_PH2_PLAN.md
```

## Stop conditions

Stop and request approval if:

- the official download cannot be accessed under its terms;
- metadata is insufficient to identify the three PH² categories;
- class mapping is ambiguous;
- image/metadata correspondence fails;
- the trained HAM10000 checkpoint is unavailable;
- any attempt would require retraining or calibration.

## Current-status note

This is a historical plan document. A later secondary-mirror PH2 audit is documented in `docs/PHASE_5_PH2_EXTERNAL_VALIDATION_REPORT.md` and summarized in `docs/PROJECT_DOCUMENTATION.md`. PH2 inference is still blocked and no PH2 metrics exist.
