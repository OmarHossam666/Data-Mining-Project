"""
Phase 1 — Data Extraction with Schema Validation.

Loads raw CSV datasets from ``data/raw/`` and validates each one against
its :class:`DatasetManifest`.  Returns a dictionary of validated DataFrames.

Usage:
    # From the pipeline orchestrator
    from etl.extract import extract_data
    datasets = extract_data()

    # Standalone
    python src/etl/extract.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Manifest — single source of truth for schema & integrity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    """Declarative specification for a raw dataset."""

    name: str
    filename: str
    min_rows: int
    required_columns: tuple[str, ...]
    description: str = ""


MANIFESTS: dict[str, DatasetManifest] = {
    "uci_student": DatasetManifest(
        name="uci_student",
        filename="uci_student_performance.csv",
        min_rows=600,
        required_columns=(
            "sex", "age", "address", "studytime", "failures",
            "paid", "internet", "absences", "G1", "G2", "G3",
        ),
        description="UCI ML Repo: Student Performance (Cortez & Silva, 2008)",
    ),
    "uci_dropout": DatasetManifest(
        name="uci_dropout",
        filename="uci_dropout_success.csv",
        min_rows=4000,
        required_columns=(
            "Target", "Age at enrollment", "Gender",
            "Curricular units 2nd sem (grade)", "Scholarship holder",
        ),
        description="UCI ML Repo: Dropout & Academic Success (Realinho et al., 2022)",
    ),
    "kaggle": DatasetManifest(
        name="kaggle",
        filename="kaggle_student_performance.csv",
        min_rows=2000,
        required_columns=(
            "Age", "Gender", "StudyTimeWeekly", "Absences",
            "GPA", "GradeClass",
        ),
        description="Kaggle: Student Performance Factors (El Kharoua, 2024)",
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when a raw dataset fails schema or integrity checks."""


def _validate_schema(df: pd.DataFrame, manifest: DatasetManifest) -> None:
    """Assert *df* has required columns and minimum rows."""
    missing = set(manifest.required_columns) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"[{manifest.name}] Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns.tolist())}"
        )
    if len(df) < manifest.min_rows:
        raise DataValidationError(
            f"[{manifest.name}] Expected ≥{manifest.min_rows} rows, got {len(df)}."
        )


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_data(
    raw_dir: Path | None = None,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """Extract raw datasets from CSV files with optional schema validation.

    Args:
        raw_dir: Override directory containing raw CSVs.
        validate: If True, validate each dataset against its manifest.

    Returns:
        Dictionary mapping dataset name to raw DataFrame.

    Raises:
        FileNotFoundError: If a CSV is missing.
        DataValidationError: If schema validation fails.
    """
    data_dir = raw_dir or config.RAW_DATA_DIR
    logger.info("Starting extraction phase from %s ...", data_dir)

    datasets: dict[str, pd.DataFrame] = {}

    for name, manifest in MANIFESTS.items():
        csv_path = data_dir / manifest.filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"[{name}] Raw file not found: {csv_path}. "
                "Run `python src/etl/download.py` to acquire datasets."
            )

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.ParserError as exc:
            logger.error("Failed to parse %s: %s", csv_path, exc)
            raise

        if validate:
            _validate_schema(df, manifest)

        datasets[name] = df
        logger.info(
            "Extracted %-15s | %5d rows × %2d cols | %s",
            name, df.shape[0], df.shape[1], manifest.description,
        )

    logger.info("Extraction complete — %d datasets loaded.", len(datasets))
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data = extract_data()
    for ds_name, df in data.items():
        print(f"\n{ds_name}: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Nulls:   {df.isnull().sum().sum()}")