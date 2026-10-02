# Fixed PH² case evidence cards

All 12 cases were fixed before Stage 6. Attention text is copied from saved human review; no new map interpretation was performed.

## IMD003

Ground truth: nv; prediction: nv; correct.

Confidence: 0.8191; entropy: 0.6604 nats; margin: 0.7172; band: HIGH.

Image-quality evidence (saved PH² domain descriptors): brightness=44.571, contrast=44.312, sharpness=78.164, saturation=0.315, dark_pixel_ratio=0.562, bright_pixel_ratio=0.014, entropy=6.450, illumination_variation=0.907.

Grad-CAM evidence: Mainly within the visible lesion-like region, with two compact hotspots across its central area. Border: Moderate; activation spans central subregions and approaches part of the feature margin. Background: Minimal; the strongest foci align with the central visible feature. Predicted versus true map: Identical/reused because true class equals predicted class; no target-class difference is assessable.

Human review: category **correct nevus**. Two localized hotspots overlap the visible central feature. This is an exploratory visual judgment, not a localization score. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: reviewed Grad-CAM case category.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD009

Ground truth: nv; prediction: nv; correct.

Confidence: 0.7287; entropy: 0.8650 nats; margin: 0.5896; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=58.159, contrast=42.864, sharpness=127.984, saturation=0.467, dark_pixel_ratio=0.147, bright_pixel_ratio=0.018, entropy=6.752, illumination_variation=0.665.

Grad-CAM evidence: Mainly within the visible oval lesion-like region, with a compact central hotspot. Border: Moderate; a secondary response reaches the lower frame/edge. Background: Minimal to moderate; the central focus dominates, with some lower-edge activation. Predicted versus true map: Identical/reused because true class equals predicted class; no target-class difference is assessable.

Human review: category **correct nevus**. The strongest activation is compact and centered over the visible oval feature; weaker activation is visible near the lower image edge. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: reviewed Grad-CAM case category.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD010

Ground truth: nv; prediction: bkl; incorrect.

Confidence: 0.7382; entropy: 0.8846 nats; margin: 0.5785; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=61.276, contrast=34.821, sharpness=111.205, saturation=0.429, dark_pixel_ratio=0.030, bright_pixel_ratio=0.014, entropy=6.110, illumination_variation=0.516.

Grad-CAM evidence: Fragmented/partial overlap around the central visible feature, with separate hotspots on different sides. Border: Moderate; activation reaches adjacent sides of the central feature. Background: Moderate; some hotspots extend into nearby tissue, though the feature boundary is not annotated. Predicted versus true map: Clearly different in emphasis; predicted bkl activation favors the upper-left/central region, while true nv activation shifts toward the right-center.

Human review: category **nevus to other**. The target maps emphasize different sides of the visible central feature and nearby tissue; no causal conclusion is supported. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD020

Ground truth: nv; prediction: bkl; incorrect.

Confidence: 0.7518; entropy: 0.6327 nats; margin: 0.5225; band: HIGH.

Image-quality evidence (saved PH² domain descriptors): brightness=48.437, contrast=45.836, sharpness=81.011, saturation=0.368, dark_pixel_ratio=0.457, bright_pixel_ratio=0.019, entropy=6.430, illumination_variation=0.875.

Grad-CAM evidence: Mainly within the visible oval lesion-like feature, with a compact upper-central hotspot. Border: Minimal to moderate; the dominant focus is interior, with limited spill toward the margin. Background: Minimal; most visible activation is close to the central feature. Predicted versus true map: Similar with a modest extent difference; both maps focus on the upper-central feature, while predicted bkl activation spreads farther downward.

Human review: category **nevus to other**. Both targets emphasize a similar upper-central region, with the predicted-class map somewhat broader than the true-class map. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, high-confidence misclassification, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD035

Ground truth: nv; prediction: mel; incorrect.

Confidence: 0.5852; entropy: 0.9808 nats; margin: 0.2595; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=40.542, contrast=38.682, sharpness=57.423, saturation=0.507, dark_pixel_ratio=0.505, bright_pixel_ratio=0.012, entropy=6.290, illumination_variation=0.875.

Grad-CAM evidence: Mainly within the central visible lesion-like feature; activation follows an elongated central region. Border: Minimal; the dominant response remains away from the outer image frame. Background: Minimal; no dominant peripheral hotspot is apparent. Predicted versus true map: Similar with a difference in extent; both maps emphasize the central elongated region, while predicted mel activation is broader than true nv activation.

Human review: category **nevus to melanoma**. Predicted and true targets overlap strongly around the central feature, with a broader predicted-class response. This does not explain why the nevus was misclassified. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD045

Ground truth: nv; prediction: mel; incorrect.

Confidence: 0.5332; entropy: 0.9593 nats; margin: 0.1759; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=35.806, contrast=41.429, sharpness=61.455, saturation=0.344, dark_pixel_ratio=0.692, bright_pixel_ratio=0.014, entropy=6.019, illumination_variation=1.067.

Grad-CAM evidence: Partially to mainly overlaps the central visible feature; the predicted map is broader than the true-class map. Border: Minimal to moderate; activation remains central with some spread around the feature. Background: Minimal; no strong distant background focus is apparent. Predicted versus true map: Partially different; the predicted mel map forms a broad horizontal hotspot, while the true nv map is more crescent-shaped within an overlapping central region.

Human review: category **nevus to melanoma**. Both maps overlap the central feature but differ in shape and spatial extent; the predicted response is broader. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD058

Ground truth: mel; prediction: bkl; incorrect.

Confidence: 0.5200; entropy: 1.3232 nats; margin: 0.2609; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=103.773, contrast=52.164, sharpness=295.016, saturation=0.376, dark_pixel_ratio=0.051, bright_pixel_ratio=0.020, entropy=7.523, illumination_variation=0.441.

Grad-CAM evidence: Partially overlaps the visible central lesion-like region; target maps emphasize different subregions. Border: Moderate; hotspots occur in separated central and side portions of the visible feature. Background: Moderate; activation appears in adjacent/peripheral regions as well as over the visible feature. Predicted versus true map: Clearly different in hotspot placement; the predicted bkl map favors left/central patches, while the true melanoma map emphasizes upper-middle and lower-central patches.

Human review: category **melanoma to other**. Predicted and true targets produce distinct spatial emphasis around the visible central feature, with some overlap. No causal explanation follows from this difference. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD061

Ground truth: mel; prediction: nv; incorrect.

Confidence: 0.8215; entropy: 0.6303 nats; margin: 0.6950; band: HIGH.

Image-quality evidence (saved PH² domain descriptors): brightness=97.477, contrast=38.689, sharpness=274.876, saturation=0.336, dark_pixel_ratio=0.018, bright_pixel_ratio=0.014, entropy=7.045, illumination_variation=0.320.

Grad-CAM evidence: Weakly associated/uncertain; the strongest response tracks the lower image edge and right periphery, and lesion boundaries are difficult to distinguish. Border: Substantial; prominent activation follows the lower image border and extends up the right side. Background: Substantial peripheral activation is visible; whether it lies outside the lesion is uncertain without a mask. Predicted versus true map: Similar overall; both target maps emphasize a lower/peripheral sweep, with differences in relative hotspot intensity.

Human review: category **melanoma to nv**. Both class targets produce prominent lower-edge and peripheral activation rather than a compact central focus; the lesion-to-background relation remains uncertain. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, high-confidence misclassification, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD063

Ground truth: mel; prediction: nv; incorrect.

Confidence: 0.5247; entropy: 0.9488 nats; margin: 0.1526; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=86.032, contrast=60.905, sharpness=195.860, saturation=0.300, dark_pixel_ratio=0.263, bright_pixel_ratio=0.018, entropy=7.453, illumination_variation=0.653.

Grad-CAM evidence: Fragmented across multiple central and peripheral regions; the strongest areas are separated rather than forming one compact focus. Border: Substantial; several hotspots approach the left, right, and lower image margins. Background: Moderate to substantial; some response extends into peripheral areas, but no lesion mask is available to define the boundary. Predicted versus true map: Partially different; the predicted nv map emphasizes a left-central region and upper-right spot, while the true melanoma map has more separated right-side/lower hotspots.

Human review: category **melanoma to nv**. Both maps are fragmented, with partly overlapping peripheral foci but different hotspot placement. The visualization does not explain the classification error. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD065

Ground truth: mel; prediction: mel; correct.

Confidence: 0.4163; entropy: 1.0816 nats; margin: 0.0475; band: LOW.

Image-quality evidence (saved PH² domain descriptors): brightness=103.928, contrast=59.353, sharpness=393.085, saturation=0.230, dark_pixel_ratio=0.061, bright_pixel_ratio=0.015, entropy=7.665, illumination_variation=0.478.

Grad-CAM evidence: Partially overlaps the visible lesion; strongest hotspots are toward the lower-left lesion margin and right inner edge, while much of the central area is less active. Border: Substantial; activation approaches the lesion/image periphery at the lower-left and right edge. Background: Moderate; peripheral activation is visible beyond or near the apparent lesion boundary. Predicted versus true map: Identical/reused because true class equals predicted class; no target-class difference is assessable.

Human review: category **correct melanoma**. The correct melanoma map is not uniformly distributed over the visible lesion; hotspots favor peripheral portions. This is a visual description only. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: low-confidence correct prediction, reviewed Grad-CAM case category.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD085

Ground truth: mel; prediction: bkl; incorrect.

Confidence: 0.6556; entropy: 0.9647 nats; margin: 0.4195; band: MODERATE.

Image-quality evidence (saved PH² domain descriptors): brightness=92.083, contrast=39.308, sharpness=334.081, saturation=0.437, dark_pixel_ratio=0.017, bright_pixel_ratio=0.019, entropy=6.939, illumination_variation=0.339.

Grad-CAM evidence: Partially overlapping/uncertain; the predicted bkl map has a central hotspot, whereas the true melanoma map is weak centrally and more peripheral. Border: Substantial for the true-class map because activation appears near the lower image edge. Background: UNCERTAIN; the lower-edge response may be outside the visible lesion, but the boundary is ambiguous in this image. Predicted versus true map: Clearly different; the predicted bkl map is centered over a broad central patch, while the true melanoma map has weaker central response and lower-edge hotspots.

Human review: category **melanoma to other**. The two targets yield different spatial patterns, including peripheral true-class activation. Lesion versus background cannot be resolved reliably from this image alone. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: class-confusion failure, reviewed Grad-CAM case category, attention-localization concern.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.

## IMD168

Ground truth: mel; prediction: mel; correct.

Confidence: 0.3990; entropy: 1.5348 nats; margin: 0.2147; band: LOW.

Image-quality evidence (saved PH² domain descriptors): brightness=70.467, contrast=64.106, sharpness=130.440, saturation=0.367, dark_pixel_ratio=0.283, bright_pixel_ratio=0.026, entropy=7.306, illumination_variation=0.822.

Grad-CAM evidence: Mainly within the visible lesion-like region, with the strongest focus over its central-right portion. Border: Moderate; activation reaches an inner lesion margin but remains concentrated centrally. Background: Minimal; the strongest response is within the central visible lesion-like area. Predicted versus true map: Identical/reused because true class equals predicted class; no target-class difference is assessable.

Human review: category **correct melanoma**. A compact, strong hotspot overlaps the visible lesion-like area. The overlay alone cannot establish lesion localization accuracy. Limitation: Stage 2 review CSV records visual categories as UNCERTAIN; no lesion masks or automated localization ground truth; Grad-CAM is not causal evidence.

Failure tags: low-confidence correct prediction, reviewed Grad-CAM case category.

Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.
