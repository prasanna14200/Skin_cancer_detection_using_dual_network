# Bounded final-checkpoint Grad-CAM — prepared, not executed

The local CPU-only preparation fixed **eight IDs before viewing any heatmap** in [gradcam_selection.json](gradcam_selection.json), SHA256 `a98b3aee53e439bd2e4b9c1d67bddfb254e28166820b33c8a6296f466fb3598e`. Selection uses lexicographically first saved HAM IDs in these groups: two correct MEL, two MEL→NV, two correct NV, one correct BKL and one correct BCC. This is a deliberately small illustrative set, not a representative or diagnostic-performance sample. The target layer is `model.resnet.nonlocal3`, and the Grad-CAM target class is the saved predicted class. The script verifies frozen checkpoint/split/source hashes, saved probability agreement, finite logits/activation/gradient/map, and writes only new Stage 20 images and a manifest. It does not change model weights or Stage 16 outputs.

The first Colab run stopped during map rendering because the heatmap still required gradients when converted to NumPy. The warning about converting `p_replayed` to a scalar had the same autograd cause. No complete Grad-CAM manifest was produced. The rendering fix detaches only the already-computed probability scalar and heatmap; it does not change the model, Grad-CAM arithmetic, selected cases, or scientific configuration. The first run's partial `final_model_gradcam_output/` must remain untouched. The corrected run writes to `final_model_gradcam_output_v2/`.

Run only in Google Colab with a Tesla T4 and the existing Drive shortcut path shown by the failed execution. Ensure the updated script is present in that folder before rerunning:

```bash
%cd /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN
!python -u analysis/stage20_reliability_completion/final_model_gradcam.py \
  --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --run
```

Expected output directory: `analysis/stage20_reliability_completion/final_model_gradcam_output_v2/`, containing eight maps, eight overlays, and `gradcam_manifest.json` with `MAPS_GENERATED_REVIEW_PENDING`. **Do not call explainability complete until the output is downloaded, its hashes/provenance are audited, and maps are qualitatively reviewed with limitations.** An OOM or numerical mismatch is a diagnostic failure, not a reason to alter the frozen classifier. This is one bounded frozen-model explanation pass, not training or HAM test optimization.
