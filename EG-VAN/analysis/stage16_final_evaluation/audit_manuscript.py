"""Read-only structural and provenance checks for the final Markdown manuscript."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "manuscript/egvan_reconstruction_final.md"
SUPPLEMENT = ROOT / "manuscript/egvan_supplementary_figures.md"
old_result_tokens = ("0.822515", "0.670912", "0.579439", "0.824457",
                     "0.604687", "0.574162", "0.625000")
required_final_tokens = ("0.823471", "0.699582", "0.958206", "56/107",
                         "41 of 51", "0.275000", "11/40", "0.542813",
                         "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5")


def audit() -> dict:
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    stale = [x for x in old_result_tokens if x in main]
    missing = [x for x in required_final_tokens if x not in main]
    tables = re.findall(r"\*\*Table (\d+)\.", main)
    figures = re.findall(r"\*\*Figure (\d+)\.", main)
    supplementary = re.findall(r"\*\*Supplementary Figure (S\d+)\.", supplement)
    links = []
    for path, body in ((MAIN, main), (SUPPLEMENT, supplement)):
        for relative in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", body):
            if "://" not in relative:
                links.append((str(path), relative, (path.parent / relative).exists()))
    broken = [f"{name}: {relative}" for name, relative, exists in links if not exists]
    forbidden_claims = [x for x in ("state-of-the-art", "clinically ready", "is untouched external validation")
                        if x in main.lower()]
    required_sections = ("## Abstract", "## 1. Introduction", "## 2. Methods",
                         "## 3. Results", "## 4. Discussion", "## 5. Limitations",
                         "## 6. Conclusion")
    missing_sections = [x for x in required_sections if x not in main]
    result = {"stale_old_result_tokens": stale, "missing_final_tokens": missing,
              "tables": tables, "figures": figures, "supplementary_figures": supplementary,
              "broken_links": broken, "forbidden_unqualified_claims": forbidden_claims,
              "missing_sections": missing_sections}
    if (stale or missing or tables != ["1", "2", "3", "4"] or figures != ["1", "2", "3", "4"]
        or supplementary != ["S1", "S2", "S3", "S4"] or broken or forbidden_claims or missing_sections):
        raise ValueError(result)
    return result


if __name__ == "__main__":
    print(audit())
