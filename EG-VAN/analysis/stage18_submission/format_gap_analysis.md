# PLOS ONE format gap analysis

The frozen [master manuscript](../../manuscript/egvan_reconstruction_final.md) is preserved. This table compares it with the verified [PLOS ONE guidelines](https://journals.plos.org/plosone/s/submission-guidelines), [figures](https://journals.plos.org/plosone/s/figures), [supporting information](https://journals.plos.org/plosone/s/supporting-information), [data/code policies](https://journals.plos.org/plosone/s/data-availability) and [AI policy](https://journals.plos.org/plosone/s/ethical-publishing-practice).

| Requirement | Current master status | Required change | Risk | Action |
|---|---|---|---|---|
| Title page, short title, author affiliations, corresponding author ORCID | Only title; no author metadata | Obtain author-approved names/order/addresses, corresponding contact and ORCID; add title page to submission copy | Submission cannot be completed | Placeholder title page prepared; author input required. |
| Abstract ≤300 words, no citations | Present, under limit; acronym-heavy | Preserve numbers; review acronym expansion | Low | Submission copy retains evidence-consistent abstract. |
| Section order and acknowledgments | Core sections present; no acknowledgments | PLOS section headings/order and acknowledgments | Moderate | Separate submission Markdown prepared; author to supply acknowledgments. |
| Vancouver references in first-use order | Numbered refs, but first-use order is [2], [1], [4]... | Renumber in submission copy only, retain bibliography content; check final journal abbreviations/punctuation during conversion | Moderate citation-link risk | First-use order repaired and audited; final typography still needs author review. |
| Tables in text near first citation | Tables present; Table 1/4 textual mentions can be sharpened | Add first-citation language and preserve editable tables | Low | Submission copy adjusted. |
| Main figures as separate TIFF/EPS, captions in text | Four PNG images embedded as Markdown links; captions present | Remove embeds from submission manuscript; provide properly sized TIFF copies, Fig1–Fig4; captions in read order | Moderate production rejection if missed | Formatting-only TIFF copies prepared; recheck visual quality before upload. |
| Supporting files and captions | Four supplementary PNGs in separate Markdown sheet | Upload S1–S4 separately and list S captions at end of main manuscript | Low | Prepared S-numbered copies/captions; author to approve inclusion. |
| Manuscript DOC/DOCX/RTF or LaTeX PDF | Markdown only; Pandoc and python-docx unavailable locally | Convert clean submission Markdown with a reliable word processor; double-space; add page/continuous line numbers | **Submission blocker** | Exact conversion steps supplied; no unreliable DOCX fabricated. |
| Data availability | Local HAM/PH² prediction CSVs and public source datasets; no public archive confirmed | Author-approved minimal-data deposit/permissions and accurate statement | **Major policy blocker** | Draft statement with placeholders; no public availability claim. |
| Code/material availability | Repository is local; public release unverified | Decide public code archive/license/documentation and checkpoint policy | **Major policy blocker** | Draft statement; no false public URL. |
| Ethics/provenance | Public datasets cited; no author-institution ethics determination | Confirm image licenses, privacy and institutional ethics/waiver basis | **Major integrity blocker** | Author questionnaire. |
| AI assistance disclosure | Master does not have a dedicated disclosure | Document tool names, use and validation in Methods after author verification | Policy blocker | Explicit placeholder in submission copy. |
| Funding, conflicts, CRediT | Unknown | Supply true statements in PLOS submission fields | Submission blocker | Author checklist; do not infer “none.” |
| Cover letter | None | Separate one-page letter, with editor suggestions and prior PLOS interactions verified by author | Moderate | Draft prepared. |
| APC/funding route | Not documented | Verify exact live fee and assistance/institutional coverage | Financial decision | Author to verify; exact individual rate NOT VERIFIED in audit. |

No model, threshold, scientific result or frozen master text is changed by these format tasks. The current title is accurate. A sentence-case title variant is used only in the venue-specific copy to meet PLOS title style.
