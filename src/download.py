"""
Phase 1 — Reproducible Data Acquisition.

Downloads all three raw datasets from their canonical sources:
  1. UCI Student Performance  (ucimlrepo ID=320)
  2. UCI Dropout & Academic Success (ucimlrepo ID=697)
  3. Kaggle Student Performance  (kaggle dataset)

Usage:
    python -m src.etl.download          # from project root
    python src/etl/download.py          # direct execution
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset manifest — single source of truth for schema & integrity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    """Declarative specification for a raw dataset."""

    name: str
    filename: str
    source_type: str                       # "ucimlrepo" | "kaggle" | "url"
    source_id: str                         # UCI repo ID or Kaggle slug
    min_rows: int                          # lower-bound sanity check
    required_columns: tuple[str, ...]      # must-have columns after download
    description: str = ""

MANIFESTS: dict[str, DatasetManifest] = {
    "uci_student": DatasetManifest(
        name="uci_student",
        filename="uci_student_performance.csv",
        source_type="ucimlrepo",
        source_id="320",
        min_rows=600,
        required_columns=(
            "sex", "age", "address", "studytime", "failures",
            "paid", "internet", "absences", "G1", "G2", "G3",
        ),
        description="UCI ML Repository: Student Performance (Cortez & Silva, 2008)",
    ),
    "uci_dropout": DatasetManifest(
        name="uci_dropout",
        filename="uci_dropout_success.csv",
        source_type="ucimlrepo",
        source_id="697",
        min_rows=4000,
        required_columns=(
            "Target", "Age at enrollment", "Gender",
            "Curricular units 2nd sem (grade)", "Scholarship holder",
        ),
        description="UCI ML Repository: Predict Students' Dropout and Academic Success (Realinho et al., 2022)",
    ),
    "kaggle": DatasetManifest(
        name="kaggle",
        filename="kaggle_student_performance.csv",
        source_type="kaggle",
        source_id="rabieelkharoua/students-performance-dataset",
        min_rows=2000,
        required_columns=(
            "Age", "Gender", "StudyTimeWeekly", "Absences",
            "GPA", "GradeClass",
        ),
        description="Kaggle: Student Performance Factors (Rabie El Kharoua, 2024)",
    ),
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_ucimlrepo(manifest: DatasetManifest, output_dir: Path) -> Path:
    """Download a dataset from the UCI ML Repository using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Install ucimlrepo: pip install ucimlrepo>=0.0.3"
        ) from exc

    logger.info("Fetching %s from UCI ML Repository (ID=%s)...", manifest.name, manifest.source_id)
    dataset = fetch_ucirepo(id=int(manifest.source_id))

    # Combine features + targets into a single DataFrame
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    out_path = output_dir / manifest.filename
    df.to_csv(out_path, index=False)
    logger.info("Saved %s → %s (%d rows, %d cols)", manifest.name, out_path, len(df), len(df.columns))
    return out_path


def _download_kaggle(manifest: DatasetManifest, output_dir: Path) -> Path:
    """Download a dataset from Kaggle using the kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "Install kaggle: pip install kaggle>=1.6.0 and configure ~/.kaggle/kaggle.json"
        ) from exc

    logger.info("Fetching %s from Kaggle (%s)...", manifest.name, manifest.source_id)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(manifest.source_id, path=str(output_dir), unzip=True)

    # Kaggle datasets may have various filenames — find the CSV
    out_path = output_dir / manifest.filename
    if not out_path.exists():
        # Try to find any CSV that was downloaded
        csvs = list(output_dir.glob("*.csv"))
        if csvs:
            csvs[0].rename(out_path)
            logger.info("Renamed %s → %s", csvs[0].name, manifest.filename)
        else:
            raise FileNotFoundError(
                f"Kaggle download completed but no CSV found in {output_dir}"
            )

    logger.info("Saved %s → %s", manifest.name, out_path)
    return out_path


def download_dataset(manifest: DatasetManifest, output_dir: Path, force: bool = False) -> Path:
    """Download a single dataset according to its manifest.

    Args:
        manifest: Dataset specification.
        output_dir: Directory to save the CSV into.
        force: If True, re-download even if the file already exists.

    Returns:
        Path to the downloaded CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / manifest.filename

    if out_path.exists() and not force:
        logger.info("Skipping %s — already exists at %s", manifest.name, out_path)
        return out_path

    if manifest.source_type == "ucimlrepo":
        return _download_ucimlrepo(manifest, output_dir)
    elif manifest.source_type == "kaggle":
        return _download_kaggle(manifest, output_dir)
    else:
        raise ValueError(f"Unknown source type: {manifest.source_type}")


def download_all(output_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download all datasets defined in MANIFESTS.

    Args:
        output_dir: Directory to save CSVs.
        force: Re-download even if files exist.

    Returns:
        Dict mapping dataset name to its file path.
    """
    paths: dict[str, Path] = {}
    for name, manifest in MANIFESTS.items():
        paths[name] = download_dataset(manifest, output_dir, force=force)
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve project root from this file's location
    _project_root = Path(__file__).resolve().parent.parent.parent
    _raw_dir = _project_root / "data" / "raw"

    force_flag = "--force" in sys.argv
    download_all(_raw_dir, force=force_flag)
    logger.info("All datasets downloaded to %s", _raw_dir)
