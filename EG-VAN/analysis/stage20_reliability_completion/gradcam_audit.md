# Stage 20 frozen final-model Grad-CAM audit

**Disposition: integrity PASS; qualitative visualization complete with important limitations.** This audit used the downloaded `final_model_gradcam_output_v2/` files. The first failed rendering attempt is separate and supplies no completed map evidence. No training, local model inference, case reselection, quality threshold, uncertainty-threshold change, or Stage 16 artifact edit occurred in this audit.

## Provenance and integrity

- The on-disk recovered Stage 15 epoch-16 checkpoint independently hashes to `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`, matching the fixed selection and output manifests.
- The pre-visualization eight-case [selection manifest](gradcam_selection.json) hashes to `a98b3aee53e439bd2e4b9c1d67bddfb254e28166820b33c8a6296f466fb3598e`; the Colab output manifest records that exact hash. The output manifest SHA256 is `fe4681a20053abd3e1fc25dbfa5c861d433183ba5f12b6593fe16309107f4fa0`.
- All eight output case records match the original selection **in order**, with exactly two correct MEL, two MEL→NV, two correct NV, one correct BKL and one correct BCC. IDs, true/predicted labels, and saved predicted-class probabilities match the immutable Stage 16 HAM prediction file (SHA256 `a8d325f1b8417f148659077ec75567f49dd13034ea8c43cf4b1a87abe6611dfc`). No case replacement was found.
- Target layer was `model.resnet.nonlocal3`; the target class for each map was that case's saved predicted class. Replayed predicted-class probabilities differed from saved probabilities by less than the script's fixed `5e-4` tolerance, and predicted labels matched.
- The output directory has exactly eight grayscale `384×384` map PNGs, eight RGB `384×384` overlay PNGs and one manifest. Every output PNG and processed-source JPG SHA256 matches the manifest. All map pixels are finite 8-bit values, each map spans 0–255 and contains 256 distinct values; none is empty or constant. See [machine-readable integrity details](gradcam_integrity.json).
- `final_model_gradcam.py` loads the model in evaluation mode, computes class-score gradients with `torch.autograd.grad`, and contains no optimizer step, parameter assignment, or checkpoint-writing path. The checkpoint file hash still matches after the reported run. The output manifest does **not** record a post-run in-memory parameter hash, GPU identity, or executing script SHA256, so those details cannot be independently attested from the downloaded output alone. The user reported a Tesla T4 execution; the map/manifest evidence itself does not prove hardware identity.

## Eight-case visual review

The [first](gradcam_audit_contact_sheet_1.png) and [second](gradcam_audit_contact_sheet_2.png) contact sheets show the processed source image, overlay and grayscale map for **every** case. In the overlay, **red denotes higher Grad-CAM values and blue lower values**; in the grayscale map, white is higher. This color direction matters: most bright central image regions look blue/dark rather than red/white. “Lesion-like area” below means visually conspicuous image structure, **not** an annotated lesion mask. Confidence is the saved predicted-class probability; entropy is calculated from the seven saved probabilities in natural-log units.

| Image ID | True → predicted | Correct? | Confidence | Entropy | Qualitative attention visualization at `nonlocal3` |
|---|---|---|---:|---:|---|
| `ISIC_0024367` | MEL → MEL | Yes | 0.703172 | 0.667444 | Broad high response over the surrounding dark image area; the conspicuous central bright region is mostly low. Diffuse/background emphasis; no obvious isolated hair hotspot. |
| `ISIC_0024516` | MEL → MEL | Yes | 0.736726 | 0.613328 | High response mainly around the central bright region rather than inside it. Diffuse/background emphasis; no single obvious artifact hotspot. |
| `ISIC_0025248` | MEL → NV | No | 0.521971 | 0.827342 | Much of the conspicuous central lesion-like region is low, with broad higher response around it and some border/nearby variation. Diffuse background/edge emphasis; the map does not explain the misclassification causally. |
| `ISIC_0025472` | MEL → NV | No | 0.833353 | 0.521912 | Central bright region is predominantly low; higher response appears across the top and surrounding area. Strong background emphasis at this layer despite a relatively high wrong-class confidence. |
| `ISIC_0024308` | NV → NV | Yes | 0.941161 | 0.224323 | Elongated central bright region is low; surrounding image is broadly high. Background emphasis, not focal central-region response. |
| `ISIC_0024309` | NV → NV | Yes | 0.952790 | 0.193580 | Central bright region is mostly low; surrounding field has broad high response and scattered low patches. Diffuse/background emphasis. |
| `ISIC_0024336` | BKL → BKL | Yes | 0.464042 | 1.107597 | Conspicuous central patch is relatively low, with much of the surrounding field higher. Background emphasis; this correct prediction also has high entropy. |
| `ISIC_0024403` | BCC → BCC | Yes | 0.598196 | 0.729271 | Distinct focal high patches overlap the conspicuous central bright structures; surrounding image is mostly low. More focal/lesion-associated visually; no strong focus on the visible peripheral line artifact. |

Across these **selected eight**, seven maps show broad high response outside the visually conspicuous central region, while the BCC example is more focal over it. This is an observation about **one ResNet non-local layer**, not a measurement of whole-model attention or a claim that the model used background causally. The architecture has another branch and fusion layers; another target layer could yield different maps. The cases were deliberately stratified and lexicographically selected, not sampled to estimate a population rate. Without lesion masks, localization ground truth, expert review or perturbation tests, clinical localization correctness cannot be established. The maps do not validate the uncertain-case review rule or image-quality gate.

## Interpretation

Final-model Grad-CAM is now **implemented, executed, saved and qualitatively evaluated**, so the explainability objective is **COMPLETE WITH LIMITATION** as a research visualization. The visualizations are not uniformly lesion-focused and should be reported honestly as such. They do not upgrade the image-quality objective: its technical proxies remain descriptive and its pre-classifier accept/reject gate remains unvalidated. The frozen classification results, checkpoint and validation-derived entropy threshold are unchanged.
