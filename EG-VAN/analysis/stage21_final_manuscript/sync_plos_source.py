"""Synchronize PLOS preparation text from the reviewed master draft.

Preserves author-required metadata, disclosure, acknowledgments, and prior
supporting-information captions. No submission or external write occurs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "manuscript/egvan_reconstruction_final.md"
PLOS = ROOT / "manuscript/submission/plos_one/manuscript_source.md"
BACKUP = ROOT / "analysis/stage21_final_manuscript/plos_manuscript_source.pre_stage21.md"


def between(text: str, start: str, end: str | None = None) -> str:
    if text.count(start) != 1:
        raise ValueError(f"Expected one marker: {start}")
    tail = text.split(start, 1)[1]
    return tail.split(end, 1)[0] if end else tail


def main() -> None:
    master = MASTER.read_text(encoding="utf-8")
    prior = BACKUP.read_text(encoding="utf-8")
    if "EG-VAN+ as a Reliability-Extension Prototype" not in master.splitlines()[0]:
        raise ValueError("Master draft has unexpected title")
    if "EG-VAN+ as a Reliability-Extension Prototype" in PLOS.read_text(encoding="utf-8"):
        raise FileExistsError("PLOS source already integrated; refusing repeated generation")
    metadata = prior.split("\n", 1)[1].split("## Abstract", 1)[0].strip("\n")
    metadata = metadata.replace("EG-VAN reconstruction and melanoma follow-up",
                                "EG-VAN+ reliability-extension prototype")
    disclosure = between(prior, "### AI assistance disclosure\n", "## Results").strip()
    acknowledgments = between(prior, "## Acknowledgments\n", "## References").strip()
    support = between(prior, "## Supporting information captions\n").strip()

    lines = master.splitlines()
    title = lines[0]
    body = "\n".join(lines[1:]).lstrip("\n")
    body = re.sub(r"^\*\*Manuscript status:\*\*.*\n\n", "", body, count=1, flags=re.M)
    body = re.sub(r"^## Supplementary figures\n.*?(?=^## References\n)", "", body,
                  count=1, flags=re.M | re.S)
    body = body.replace("## 1. Introduction", "## Introduction")
    body = body.replace("## 2. Methods", "## Materials and methods")
    body = body.replace("## 3. Results", "## Results")
    body = body.replace("## 4. Discussion", "## Discussion")
    body = body.replace("## 5. Limitations", "### Limitations")
    body = body.replace("## 6. Conclusion", "## Conclusion")
    body = re.sub(r"^### [23]\.\d+ ", "### ", body, flags=re.M)
    body = re.sub(r"^!\[[^\]]*\]\([^\n]*\)\n\n", "", body, flags=re.M)
    body = re.sub(r"\*\*Figure (\d+)\.\*\*", r"**Fig \1.**", body)
    body = body.replace("## Results\n", "### AI assistance disclosure\n\n" + disclosure + "\n\n## Results\n", 1)
    body = body.replace("## References\n", "## Acknowledgments\n\n" + acknowledgments + "\n\n## References\n", 1)
    output = title + "\n\n" + metadata + "\n\n" + body.rstrip() + "\n\n## Supporting information captions\n\n" + support + "\n"
    if any(token not in output for token in ("[AUTHOR NAMES", "### AI assistance disclosure", "## Acknowledgments",
                                              "**Fig 8.**", "**Table 6.")):
        raise ValueError("PLOS preparation content missing expected section or placeholder")
    PLOS.write_text(output, encoding="utf-8")
    print("PLOS manuscript source synchronized; author placeholders preserved")


if __name__ == "__main__":
    main()
