# Convert the PLOS ONE submission source

The separate `manuscript_source.md` is an **editable preparation source**, not an upload-ready manuscript file. PLOS ONE accepts DOC/DOCX/RTF, or a PDF compiled from LaTeX; Markdown alone is not an accepted manuscript upload. Pandoc and `python-docx` were unavailable in this local environment, so no DOCX was fabricated. The frozen `manuscript/egvan_reconstruction_final.md` remains untouched.

1. After authors replace all bracketed placeholders and approve data/code/ethics/AI statements, open `manuscript_source.md` in a Markdown-aware editor or use a trusted Markdown→DOCX converter. Inspect the resulting DOCX manually. Do not upload a PDF converted from Markdown as if it were a LaTeX submission.
2. In Word or LibreOffice, use single-column text, double line spacing, page numbers, and **continuous** line numbers. Check title page, heading levels, Greek letters/PH², all four editable tables, and numbered references. The abstract is below 300 words; recheck after any author edits.
3. Upload `figures/Fig1.tif` through `Fig4.tif` individually as main figures. These are formatting-only copies from Stage 16 PNGs: RGB, LZW TIFF, 300 dpi, within PLOS width/height and file-size limits. Inspect the rendered journal preview before final confirmation. Keep the in-text `Fig 1`–`Fig 4` citations and captions; do not embed main figures in the manuscript file.
4. Upload `figures/S1_fig.png` through `S4_fig.png` individually as supporting information if authors retain them. Keep matching S1–S4 captions at the end of the manuscript. Supporting files are published as supplied, so inspect each.
5. Enter data availability, funding and competing-interest information into the required submission-system fields after author approval. Do not treat placeholder drafts as actual declarations. Provide author CRediT roles and corresponding-author ORCID.
6. Submit the separate one-page cover letter only after authors complete prior-PLOS-contact and editor-suggestion fields. Any publication-fee assistance request belongs in the submission system, not the letter.
7. Run the final scientific consistency check against the frozen master and Stage 16 manifests after DOCX/RTF conversion. Conversion must not change result numbers, PH² terminology, recovery limitation or selective-FP32 description.

Official rules: [manuscript guidelines](https://journals.plos.org/plosone/s/submission-guidelines), [main figures](https://journals.plos.org/plosone/s/figures), [supporting information](https://journals.plos.org/plosone/s/supporting-information).
