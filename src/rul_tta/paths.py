from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
CURRENT_RESULTS_DIR = RESULTS_ROOT / "current"
WORKBENCH_RESULTS_DIR = RESULTS_ROOT / "workbench"

# The repository is currently stored beside the original C-MAPSS directory.
# RUL_TTA_CMAPSS_DIR makes the package portable without hard-coding a machine.
DEFAULT_CMAPSS_DIR = Path(
    os.environ.get("RUL_TTA_CMAPSS_DIR", PROJECT_ROOT.parent / "CMAPSSData")
)
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
