# Paper preprocessing versus frozen implementation

Paper source: [Saeed et al.](https://doi.org/10.1109/ACCESS.2025.3561240), Section III-A/B and equations 1–15. Code: `src/preprocessing.py`, `src/dataset.py`, and `experiments/egvan_melanoma_ablation_exp/train_ablation.py`. Match refers to documented operations, not author-code identity. No preprocessing was changed here.

| Component | Paper | Our implementation | Match? | Potential performance impact | Evidence |
|---|---|---|---|---|---|
| Hair removal | Grayscale morphology, mask, Telea | Grayscale blackhat, threshold 10, Telea radius 1 | CLOSE | Artifact masking can alter lesion texture; direction unknown | Paper eqs. 1–7; `remove_hair` |
| Black-hat filtering | Prose says “Black top-hat”; printed `g - opening(g)` is bright top-hat | `cv2.MORPH_BLACKHAT` = closing minus image, ellipse 17×17 | PAPER AMBIGUOUS / DIFFERENT from printed equation | Could change hair mask; no measured benefit | Paper eq. 5; `HAIR_KERNEL_SIZE` |
| Thresholding | Binary mask with threshold T; T **NOT SPECIFIED IN PAPER** | Fixed 10, binary | CLOSE | Mask density may affect detail | Paper eq. 6; `HAIR_THRESHOLD` |
| Telea inpainting | Explicit | `cv2.INPAINT_TELEA`, radius 1 | CLOSE | Radius unreported by paper | Paper text; `INPAINT_RADIUS` |
| Gray World | Channel-mean gain correction | Per-image BGR gains, clipping to uint8 | CLOSE | Color distribution can change | Paper eqs. 8–9; `gray_world` |
| Retinex | Weighted Gaussian/log multi-scale response | Sigmas 15/80/250, equal weights; per-channel percentile rescaling | CLOSE | Scale/normalization may alter contrast | Paper eqs. 10–13; `retinex` |
| Combined Gray World + Retinex | Explicit combination | `remove_hair → gray_world → retinex` | CLOSE | Direction and magnitude untested as isolated factor | Paper Fig. 2; `preprocess_image` |
| Cropping | Square crop expression; center/size **NOT SPECIFIED IN PAPER** | No crop in frozen processed cache | MISSING | Removing background might help or remove lesion context; untested | Paper eq. 14; preprocessing code |
| Resizing | Set image to model size; size **NOT SPECIFIED IN PAPER** | 384×384 | PAPER AMBIGUOUS | Scale/context changes could matter | Paper Section III-A; Stage15 config |
| Normalization | **NOT SPECIFIED IN PAPER** | ImageNet mean/std after RGB conversion | PAPER AMBIGUOUS | Affects pretrained branch input statistics | `src/dataset.py` / train transforms |
| Augmentation | 20 transformations mentioned; exact operators **NOT SPECIFIED IN PAPER** | On-the-fly train flips/rotation; validation unaugmented | DIFFERENT / PAPER AMBIGUOUS | Regularization and apparent validation difficulty differ | Paper eq. 15 and Section IV-C; train loader |

The paper’s numerical pipeline cannot be reproduced exactly from the publication. Our preprocessing is paper-informed and already includes most named operations. Missing crop and non-identical augmentation are concrete gaps, but neither has demonstrated accuracy benefit. Any change needs a separately registered train/validation-only comparison using unchanged lesion-isolated partitions; old checkpoint predictions cannot be used to infer its effect.
