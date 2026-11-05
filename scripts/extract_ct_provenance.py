#!/usr/bin/env python
"""Command-line entry point for ct_dicom_conversion."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ct_dicom_conversion import main

if __name__ == "__main__":
    raise SystemExit(main())
