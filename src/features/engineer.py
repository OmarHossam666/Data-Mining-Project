"""
Phase 5 — Feature Engineering with Temporal Checkpoint Simulation.

Reads full-fidelity processed CSVs (not the narrow warehouse schema) and
creates checkpoint-specific train/val/test splits using **feature masking**
to simulate temporal information availability.

Checkpoint Design (for UCI Dropout dataset — primary modelling dataset):
  - **Week 4 (Enrollment)**: Only demographic + application features
    available at the time of enrolment (21 features).
  - **Week 8 (Mid-Semester)**: Enrollment features + 1st semester academic
    performance becomes visible (27 features).
  - **Week 12 (Late-Semester)**: All features including 2nd semester
    performance and macroeconomic indicators (36 features).

Key design decisions:
  - Primary model trained on UCI Dropout (n=4,424, richest feature set).
  - Cross-dataset validation on UCI Student + Kaggle using shared features.
  - SMOTE applied ONLY to training split.
  - StandardScaler fit ONLY on training split.
  - Target-source columns (``Target``) excluded from features.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Output directories
FEATURES_DIR = config.FEATURES_DIR
RESULTS_DIR = config.RESULTS_DIR

# ---------------------------------------------------------------------------
# Checkpoint Feature Masks (UCI Dropout temporal simulation)
# ---------------------------------------------------------------------------
# These define which columns are AVAILABLE at each checkpoint.  This simulates
# a real early-warning system where predictions must be made with incomplete
# information.

# Columns that MUST NEVER be used as features (target-source leakage)
LEAKAGE_COLUMNS: set[str] = {"Target", "risk_label"}

# --- Features available at enrollment (Week 4) ---
FEATURES_ENROLLMENT: list[str] = [
    "Marital Status", "Application mode", "Application order", "subject",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "nationality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "sex", "scholarship_flag",
    "age", "International",
]

# --- Features first available at mid-semester (Week 8) ---
FEATURES_SEM1: list[str] = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

# --- Features first available at late-semester (Week 12) ---
FEATURES_SEM2: list[str] = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "gpa",    # = "Curricular units 2nd sem (grade)", renamed in transform
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]

CHECKPOINT_FEATURE_SETS: dict[int, list[str]] = {
    4:  FEATURES_ENROLLMENT,
    8:  FEATURES_ENROLLMENT + FEATURES_SEM1,
    12: FEATURES_ENROLLMENT + FEATURES_SEM1 + FEATURES_SEM2,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data(dataset_name: str = "uci_dropout") -> pd.DataFrame:
    """Load a full-fidelity processed CSV from data/processed/.

    Args:
        dataset_name: One of 'uci_student', 'uci_dropout', 'kaggle'.

    Returns:
        Cleaned DataFrame with all original columns preserved.
    """
    csv_path = config.PROCESSED_DATA_DIR / f"{dataset_name}_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {csv_path}. "
            "Run the ETL pipeline first (etl_pipeline.py)."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows × %d cols", dataset_name, *df.shape)
    return df


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def generate_correlation_heatmap(df: pd.DataFrame, save_path: Path | None = None) -> pd.DataFrame:
    """Generate and save a Pearson correlation heatmap for numeric features."""
    logger.info("Generating feature correlation heatmap ...")
    numeric_df = df.select_dtypes(include=[np.number])

    # Drop ID and target columns from correlation
    drop_cols = [c for c in ("fact_id", "risk_label", "StudentID") if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=drop_cols, errors="ignore")

    corr = numeric_df.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (UCI Dropout)")
    plt.tight_layout()
    out = save_path or (RESULTS_DIR / "feature_correlation_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved correlation heatmap → %s", out)
    return corr


def remove_collinear_features(
    df: pd.DataFrame,
    threshold: float = config.COLLINEARITY_THRESHOLD,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one of each pair of features with |r| > threshold.

    Returns:
        Tuple of (reduced DataFrame, list of dropped column names).
    """
    exclude = set(exclude_cols or [])
    numeric = df.select_dtypes(include=[np.number]).drop(columns=list(exclude), errors="ignore")
    corr_abs = numeric.corr().abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    logger.info("Collinearity removal (r > %.2f): dropping %d features: %s",
                threshold, len(to_drop), to_drop)
    return df.drop(columns=to_drop), to_drop


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------

def process_checkpoint(
    df: pd.DataFrame,
    checkpoint_week: int,
    name: str,
    top_k: int = config.TOP_K_FEATURES,
) -> dict[str, pd.DataFrame] | None:
    """Process one checkpoint: select features → split → SMOTE → scale → save.

    Args:
        df: Full processed DataFrame (all columns present).
        checkpoint_week: Which checkpoint to simulate (4, 8, or 12).
        name: Identifier string like 'checkpoint_4'.
        top_k: Maximum number of features to select.

    Returns:
        Dict with 'train', 'val', 'test' DataFrames, or None if no data.
    """
    logger.info("=" * 60)
    logger.info("Processing %s (Week %d) ...", name, checkpoint_week)

    feature_cols = CHECKPOINT_FEATURE_SETS[checkpoint_week]

    # Keep only columns that exist in the DataFrame
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("[%s] %d expected features not found: %s", name, len(missing), missing)

    if not available:
        logger.error("[%s] No features available — skipping.", name)
        return None

    X = df[available].copy()
    y = df["risk_label"].copy()

    # Drop rows without target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    logger.info("[%s] Feature matrix: %d rows × %d features", name, *X.shape)
    logger.info("[%s] Class distribution: %s", name, y.value_counts().to_dict())

    # One-hot encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info("[%s] After one-hot encoding: %d features", name, X.shape[1])

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=["boolean", "bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # Fill remaining NaN with column median (not 0)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # ── Train / Val / Test split (70 / 15 / 15) ─────────────────────────
    # The split is performed BEFORE feature selection to prevent any
    # information from the test set from influencing the MI scores.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_TEST_SPLIT,
        stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )

    logger.info("[%s] Split sizes — train: %d, val: %d, test: %d",
                name, len(X_train), len(X_val), len(X_test))

    # ── Feature selection on TRAINING data only (leakage-free) ────────
    # Mutual Information is computed exclusively on the training partition.
    # Validation and test sets are then subsetted to the same features.
    k = min(top_k, X_train.shape[1])
    logger.info("[%s] Selecting top %d features via Mutual Information (training data only) ...", name, k)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_SEED)
    mi_ranking = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    selected_features = mi_ranking.head(k).index.tolist()
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    logger.info("[%s] Selected features: %s", name, selected_features)

    # ── SMOTE on training set ONLY ────────────────────────────────────
    logger.info("[%s] Before SMOTE: %s", name, pd.Series(y_train).value_counts().to_dict())
    smote = SMOTE(k_neighbors=5, random_state=config.RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info("[%s] After  SMOTE: %s", name, pd.Series(y_train_res).value_counts().to_dict())

    # ── StandardScaler fit on train ONLY ──────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_res), columns=selected_features
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=selected_features
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=selected_features
    )

    # Save scaler artifact
    scaler_path = config.MODELS_DIR / f"scaler_{name}.joblib"
    joblib.dump(scaler, scaler_path)

    # Save datasets with target column
    train_out = X_train_scaled.copy()
    train_out["target"] = y_train_res.values

    val_out = X_val_scaled.copy()
    val_out["target"] = y_val.values

    test_out = X_test_scaled.copy()
    test_out["target"] = y_test.values

    train_out.to_csv(FEATURES_DIR / f"{name}_train.csv", index=False)
    val_out.to_csv(FEATURES_DIR / f"{name}_val.csv", index=False)
    test_out.to_csv(FEATURES_DIR / f"{name}_test.csv", index=False)

    logger.info("[%s] Saved train/val/test CSVs to %s", name, FEATURES_DIR)
    return {"train": train_out, "val": val_out, "test": test_out}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_engineering() -> None:
    """Execute the full feature engineering pipeline."""
    # 1. Load primary dataset (UCI Dropout — richest features + temporal structure)
    df = load_processed_data("uci_dropout")

    # 2. Correlation analysis
    generate_correlation_heatmap(df)

    # 3. Process each checkpoint with feature masking
    for week in config.CHECKPOINT_WEEKS:
        name = f"checkpoint_{week}"
        process_checkpoint(df, checkpoint_week=week, name=name)

    logger.info("=" * 60)
    logger.info("Phase 5 Feature Engineering complete!")


if __name__ == "__main__":
    run_feature_engineering()