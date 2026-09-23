#!/usr/bin/env python3
"""Phase 4A entry point for metadata validation."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "prepare_metadata.py"), run_name="__main__")
