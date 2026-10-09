"""Local reviewer interface for the frozen Stage 23 EG-VAN+ research model."""
from __future__ import annotations

import io
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import analyze_image, load_pipeline  # noqa: E402

st.set_page_config(page_title="EG-VAN+ research viewer", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Verifying and loading frozen Stage 23 model...")
def _model():
    return load_pipeline()


def main() -> None:
    st.title("EG-VAN+ skin lesion research viewer")
    st.caption("Frozen Stage 23 epoch-14 classifier · seven HAM10000 classes · local inference")
    st.warning("Research use only. This output is not a medical diagnosis or clinical decision aid. Seek a qualified clinician for assessment.")
    uploaded = st.file_uploader("Upload a lesion image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a PNG or JPEG image to view the model output.")
        return
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        if image.width * image.height > 30_000_000:
            raise ValueError("Image exceeds 30 million pixels")
        image.load()
        original = image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not open image: {exc}")
        return
    show_cam = st.checkbox("Generate Grad-CAM attention visualization (slower)", value=False)
    if show_cam:
        st.caption("Full-resolution Grad-CAM may require substantially more memory than prediction, especially on CPU.")
    col_image, col_result = st.columns([1, 1])
    with col_image:
        st.image(original, caption="Original uploaded image", use_container_width=True)
    try:
        model, transform, classes = _model()
        with st.spinner("Analyzing image..."):
            result = analyze_image(original, model, transform, classes, include_gradcam=show_cam)
    except Exception as exc:
        st.error(f"Inference unavailable: {type(exc).__name__}: {exc}")
        return
    with col_result:
        st.subheader(f"Predicted class: {result['predicted_class'].upper()}")
        st.metric("Top softmax score", f"{result['confidence']:.3f}")
        st.metric("Predictive entropy", f"{result['predictive_entropy']:.3f}")
        st.caption(f"Normalized entropy: {result['normalized_predictive_entropy']:.3f} (0 to 1)")
        st.bar_chart(pd.Series(result["probabilities"], name="Softmax score"), horizontal=True)
        st.info("No Stage 23 review cutoff is validated; entropy is shown without an accept/review decision.")
        st.caption("Softmax scores are not calibrated clinical probabilities.")
    with st.expander("Technical image-quality measurements", expanded=True):
        st.caption("Measured on the resized 384×384 RGB view. These are descriptive proxies, not a validated image acceptance gate.")
        labels = {"brightness": "Brightness (0–255)", "contrast": "Contrast (gray SD)",
                  "sharpness": "Laplacian variance", "saturation": "Mean saturation (0–1)",
                  "dark_pixel_ratio": "Dark pixel fraction (<30)",
                  "bright_pixel_ratio": "Bright pixel fraction (>225)",
                  "entropy": "Gray histogram entropy (bits)",
                  "illumination_variation": "Spatial illumination variation"}
        st.dataframe({"Indicator": [labels[k] for k in result["image_quality"]],
                      "Value": [round(v, 4) for v in result["image_quality"].values()]},
                     hide_index=True, use_container_width=True)
    if show_cam and result["gradcam_overlay"] is not None:
        st.subheader("Qualitative model-attention visualization")
        if result["gradcam_status"].startswith("Degenerate"):
            st.warning(result["gradcam_status"])
        st.image(result["gradcam_overlay"], caption="Predicted-class Grad-CAM over the resized model input", use_container_width=True)
        st.caption("Highlighted areas do not establish a correct diagnosis or clinically validated localization.")
    with st.expander("Model provenance"):
        st.write(f"Model: `{result['model_identifier']}` · selected epoch: {result['selected_epoch']}")
        st.code(result["checkpoint_sha256"])
        st.caption("Stage 23 was selected on validation data. Earlier Stage 15 test/PH2 results do not measure this classifier.")


if __name__ == "__main__":
    main()
