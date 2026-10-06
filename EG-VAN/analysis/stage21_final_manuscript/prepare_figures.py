"""Prepare manuscript-only figure assets from audited Stage 20 evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis/stage21_final_manuscript"
STAGE20 = ROOT / "analysis/stage20_reliability_completion"
PLOS = ROOT / "manuscript/submission/plos_one/figures"


def make_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 10.6))
    ax.set(xlim=(0, 10), ylim=(0, 15))
    ax.axis("off")

    def box(y: float, label: str, color: str, width: float = 7.2, height: float = 1.02) -> None:
        left = (10 - width) / 2
        ax.add_patch(FancyBboxPatch((left, y), width, height,
                                    boxstyle="round,pad=0.12,rounding_size=0.18",
                                    linewidth=1.2, edgecolor="#334155", facecolor=color))
        ax.text(5, y + height / 2, label, ha="center", va="center", fontsize=10.5, color="#17212f")

    def down(top: float, bottom: float) -> None:
        ax.add_patch(FancyArrowPatch((5, top), (5, bottom), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.1, color="#334155"))

    verified, partial = "#e3f3ed", "#fff1cf"
    box(13.6, "Dermoscopic image", verified)
    down(13.55, 13.1)
    box(12.0, "Technical image-quality risk proxies\nPARTIAL: no validated accept/reject gate", partial)
    down(11.95, 11.5)
    box(10.4, "Frozen preprocessing: hair removal, color balancing", verified)
    down(10.35, 9.9)
    box(8.8, "Frozen EG-VAN: EfficientNetV2S + modified ResNet50\nSCGA / Non-Local + multiscale fusion", verified)
    down(8.75, 8.3)
    box(7.2, "Seven-class probability vector + frozen argmax", verified)
    down(7.15, 6.7)
    box(5.6, "Predictive entropy\nvalidation-derived threshold 0.7675495327940953", verified)
    down(5.55, 5.1)
    ax.add_patch(FancyArrowPatch((5, 5.1), (2.5, 4.6), arrowstyle="-|>", mutation_scale=12, color="#334155"))
    ax.add_patch(FancyArrowPatch((5, 5.1), (7.5, 4.6), arrowstyle="-|>", mutation_scale=12, color="#334155"))
    for x, label in ((0.6, "Entropy ≤ threshold\nretain prediction"),
                     (5.6, "Entropy > threshold\nflag for review")):
        ax.add_patch(FancyBboxPatch((x, 3.4), 3.8, 1.12,
                                    boxstyle="round,pad=0.12,rounding_size=0.18",
                                    linewidth=1.2, edgecolor="#334155", facecolor=verified))
        ax.text(x + 1.9, 3.96, label, ha="center", va="center", fontsize=10.2)
    ax.text(5, 2.75, "Bounded one-layer Grad-CAM: qualitative, eight fixed cases",
            ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=partial, edgecolor="#334155"))
    ax.text(5, 1.75, "Evaluation: HAM held-out test + PH² external follow-up",
            ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=verified, edgecolor="#334155"))
    ax.text(5, .85, "Prototype reliability layer; full-cohort classifier accuracy is unchanged",
            ha="center", va="center", fontsize=9.7, color="#334155")
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    pipeline = ANALYSIS / "egvan_plus_pipeline.png"
    if pipeline.exists():
        raise FileExistsError(pipeline)
    make_pipeline(pipeline)
    sources = {
        "Fig5.tif": STAGE20 / "ham_reliability_diagram.png",
        "Fig6.tif": STAGE20 / "ham_risk_coverage.png",
        "Fig7.tif": STAGE20 / "gradcam_audit_contact_sheet_1.png",
        "Fig8.tif": pipeline,
    }
    for name, source in sources.items():
        target = PLOS / name
        if target.exists():
            raise FileExistsError(target)
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="TIFF", dpi=(350, 350), compression="tiff_lzw")
    print("Created pipeline PNG and PLOS Fig5–Fig8 TIFF files from audited inputs")


if __name__ == "__main__":
    main()
