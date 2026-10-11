# Stage 4D v2 split integrity audit

**Independent read-only result: `PASS_STAGE4D_V2_INTEGRITY`.** The [auditor](audit_stage4d_v2.py) rehashed all 8,015 original raw and processed training images, v1/v2 manifests and generated CSVs; independently recomputed lesion assignment and the full cross-boundary similarity screen by direct pair comparison. The v2 manifest SHA256 is `e218c9c2772452738cc53753d3b8857f1d321722a3e64b0b0e389442fcdadc93`. This is an integrity result, **not** human adjudication or proof of patient independence.

## Inputs, graph and assignment

Stage 4B v1 manifest SHA256 remained `dd2ce3f2d698ff5cbf392ccc2a8889338fe4cb7b1017dcff677214eec392e649`; original HAM leakage-aware split SHA256 remained `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The 40 Stage 4C refined image-pair edges join 59 lesion IDs in 20 components. They are *possible* similarities, with no filled reviewer decision. Every image of each originally affected lesion was included in graph-based accounting.

The preregistered v2 rule moved all **23 original selection lesions** touching those components into training. A pre-build screen showed that this direct move would create two newly cross-boundary refined pairs among originally selection-side lesions. The fixed graph-closure rule therefore moved **two additional selection lesions**. Altogether 25 entire selection lesions / 52 images moved to development training. No images were discarded or reassigned by diagnosis, appearance judgment, prediction or model outcome.

| Role | Images | Lesions | MEL images | MEL lesions |
|---|---:|---:|---:|---:|
| Stage 4D development training | 6,477 | 4,804 | 736 | 396 |
| Stage 4D internal selection | 1,538 | 1,169 | 163 | 95 |

See the complete seven-class [distribution CSV](STAGE4D_CLASS_DISTRIBUTION.csv). All seven classes remain in each role. The 163 selection melanoma images / 95 lesions support *exploratory* MEL recall estimation, with lesion-group uncertainty; they do not provide patient-group uncertainty.

## Integrity checks

- Exactly 8,015 original leakage-aware HAM training images occur once across the two v2 manifests. No original HAM validation or test image/lesion occurs. PH2 is absent.
- No lesion ID crosses training and selection; no identical raw-JPEG **or** processed-JPEG byte hash crosses. All 40 originally refined Stage 4C pairs now have both endpoints in development training.
- Recomputing the original within-selection two-stage screen found 669 dHash≤4 pairs, of which 3 also met pHash≤8. The closure contains every refined edge incident to a moved lesion. The third within-selection refined edge remains inside one role and does not cross the boundary.
- Full v2 cross-boundary re-screen found **5,189 dHash≤4 pairs**, but **zero** also met the fixed pHash≤8 refinement criterion. Every pair and score is preserved in [boundary screen CSV](stage4d_boundary_screen_v2.csv). These 5,189 low-resolution screen matches are not declared unrelated or independently adjudicated.
- All source/image, protocol/script and output hashes matched the [v2 manifest](stage4d_split_manifest_v2.json). The frozen Stage 23 checkpoint SHA256 remained `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab` in a read-only hash check; it was never loaded for inference.

## Versioned output hashes

| Artifact | SHA256 |
|---|---|
| [Development train v2](stage4d_development_train_v2.csv) | `3adc5f781cd0ff4641f1271064e9e24c83f5a045e4e969c1a7c1aecaab91905f` |
| [Internal selection v2](stage4d_internal_selection_v2.csv) | `fae3fd546267226df8211828dca791656bb384470ca06a7d99229dca198429d8` |
| [Class distribution](STAGE4D_CLASS_DISTRIBUTION.csv) | `5863372213edf14aba1424c701b030594d873cef3a5ceedcb5b8644a08a511ea` |
| [Cross-boundary screen](stage4d_boundary_screen_v2.csv) | `6bdc68f5d53db84cf3484b3194e530a94edaaa4273330eb0b5de6449608f5a41` |

No training, model inference, threshold tuning, human adjudication, original partition edit or frozen artifact edit occurred. The fresh-control and single Gaussian-blur treatment protocol is unchanged; any future models must both use these **same v2 manifests**.
