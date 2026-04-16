"""
ProGrade Project Configuration.

Central configuration module providing path constants, database URI,
and logging setup. All paths are resolved relative to the project root
(determined by this file's location) for portability across machines.
"""

from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
GOLD_DATA_DIR: Path = DATA_DIR / "gold"
FEATURES_DIR: Path = PROJECT_ROOT / "features"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Ensure directories exist on import
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, GOLD_DATA_DIR,
             FEATURES_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------

DB_PATH: Path = PROJECT_ROOT / "warehouse.db"
DATABASE_URI: str = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
BATCH_SIZE: int = 500
TEST_SIZE: float = 0.30
VAL_TEST_SPLIT: float = 0.50          # splits the 30% temp into 15%/15%
TOP_K_FEATURES: int = 15
COLLINEARITY_THRESHOLD: float = 0.85
CHECKPOINT_WEEKS: tuple[int, ...] = (4, 8, 12)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("Project configuration loaded — root: %s", PROJECT_ROOT)