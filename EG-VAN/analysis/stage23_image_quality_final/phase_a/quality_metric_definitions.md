# Stage 23 technical image measurements

Source: [`src/image_quality.py`](../../../src/image_quality.py). OpenCV converts RGB to 8-bit grayscale and HSV. Every feature was measured on (1) raw 600×450 JPEG, (2) paper-preprocessed 600×450 JPEG, and (3) processed RGB image bilinearly resized to 384×384 before tensor normalization. The analysis uses **raw** values as the primary predictors; processed and resized values remain in the historical table for auditing. Resize and preprocessing can substantially alter the values.

| Feature | Exact calculation | Units/range | Higher value | Main confounders and construct validity |
|---|---|---|---|---|
| Brightness | Mean grayscale intensity | 8-bit intensity, 0–255 | Brighter, not inherently better | Skin tone, illumination, lesion pigment, preprocessing; an image characteristic, not an exposure-failure label. |
| Contrast | Population SD of grayscale pixels | Intensity units, approximately 0–127.5 | More global tonal variation, not inherently better | Lesion pigmentation and size, framing, illumination; not a verified contrast-artifact detector. |
| Sharpness | Population variance of OpenCV 64-bit Laplacian of grayscale | Squared Laplacian response, unbounded nonnegative | More high-frequency edges, not necessarily better focus | Hair, rulers, noise, lesion texture, JPEG edges, resolution; blur proxy only. |
| Saturation | Mean OpenCV HSV S channel divided by 255 | Fraction, 0–1 | More saturated color, not inherently better | Skin/lesion color, lighting, white balance, processing; not a quality label. |
| Dark-pixel ratio | Fraction of grayscale pixels strictly below 30 | Fraction, 0–1 | More dark pixels, not inherently worse | Dark lesions, borders, vignette, skin tone; 30 is a counting threshold, not an image-rejection rule. |
| Bright-pixel ratio | Fraction of grayscale pixels strictly above 225 | Fraction, 0–1 | More bright pixels, not inherently worse | Pale skin, glare, background, lighting; 225 is a counting threshold, not an image-rejection rule. |
| Histogram entropy | −Σ p(i) log₂ p(i) over nonzero 8-bit grayscale bins | Bits, 0–8 | More tonal diversity, not inherently better | Texture, lesion color, illumination, compression; not predictive entropy or a direct artifact measure. |
| Illumination variation | SD of mean grayscale intensities of 16×16 spatial cells divided by mean of those cell means; zero if denominator zero | Unitless, ≥0 | More spatial variation, not inherently worse | Lesion position/size, vignetting, hair, shadows, uneven lighting; does not isolate illumination artifacts. |

These deterministic proxies cannot establish that a dark, low-saturation, highly textured, or low-texture lesion image is technically inadequate. No clinical accept/reject threshold follows from these definitions.
