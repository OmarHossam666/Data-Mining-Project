"""
Phase 8 — SHAP Explainability.

Generates SHAP-based explanations for the best model (XGBoost, Checkpoint 12):
  - Global feature importance (bar + beeswarm summary plots)
  - Individual student waterfall plots for different risk tiers
  - Advisor intervention reports (PDF)

Usage:
    python src/run_shap.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config
from explain.shap_reporter import generate_advisor_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESULTS_DIR = config.RESULTS_DIR


def load_model_and_data() -> tuple:
    """Load the best model (XGBoost CP12) and its test data."""
    model = joblib.load(config.MODELS_DIR / "xgb_model.joblib")
    test_df = pd.read_csv(config.FEATURES_DIR / "checkpoint_12_test.csv")
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(int)
    logger.info("Loaded XGBoost model and test data: %d samples × %d features",
                *X_test.shape)
    return model, X_test, y_test


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer."""
    logger.info("Computing SHAP values with TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("SHAP computation complete. Shape: %s", shap_values.shape)
    return shap_values


def plot_summary_bar(shap_values: shap.Explanation, save_path: Path) -> None:
    """Global feature importance — mean |SHAP| bar chart."""
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP bar plot → %s", save_path)


def plot_beeswarm(shap_values: shap.Explanation, save_path: Path) -> None:
    """SHAP beeswarm summary — shows feature value impact direction."""
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary — Feature Impact on Risk Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm → %s", save_path)


def plot_waterfall(
    shap_values: shap.Explanation,
    idx: int,
    title: str,
    save_path: Path,
) -> None:
    """Individual student waterfall plot."""
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved waterfall (%s) → %s", title, save_path)


def generate_waterfall_gallery(
    model, X: pd.DataFrame, y: pd.Series, shap_values: shap.Explanation,
) -> None:
    """Generate waterfall plots for representative students."""
    y_prob = model.predict_proba(X)[:, 1]

    # High risk: predicted high, actually high, highest probability
    high_risk_mask = (y == 1)
    if high_risk_mask.any():
        idx = y_prob[high_risk_mask].argmax()
        actual_idx = X.index[high_risk_mask][idx]
        shap_idx = list(X.index).index(actual_idx)
        plot_waterfall(
            shap_values, shap_idx,
            f"High-Risk Student (P={y_prob[high_risk_mask][idx]:.2f})",
            RESULTS_DIR / "shap_waterfall_high_risk.png",
        )

    # Low risk: predicted low, actually low, lowest probability
    low_risk_mask = (y == 0)
    if low_risk_mask.any():
        idx = y_prob[low_risk_mask].argmin()
        actual_idx = X.index[low_risk_mask][idx]
        shap_idx = list(X.index).index(actual_idx)
        plot_waterfall(
            shap_values, shap_idx,
            f"Low-Risk Student (P={y_prob[low_risk_mask][idx]:.2f})",
            RESULTS_DIR / "shap_waterfall_low_risk.png",
        )

    # Medium risk: probability closest to 0.5
    mid_idx = (np.abs(y_prob - 0.5)).argmin()
    shap_mid_idx = list(X.index).index(X.index[mid_idx])
    plot_waterfall(
        shap_values, shap_mid_idx,
        f"Borderline Student (P={y_prob[mid_idx]:.2f})",
        RESULTS_DIR / "shap_waterfall_medium_risk.png",
    )

    # Generate PDF report for the highest-risk student
    if high_risk_mask.any():
        idx = y_prob[high_risk_mask].argmax()
        actual_idx = X.index[high_risk_mask][idx]
        shap_idx = list(X.index).index(actual_idx)
        sv = shap_values[shap_idx]

        feature_impacts = list(zip(sv.feature_names, sv.values))
        positive = [(f, v) for f, v in feature_impacts if v > 0]
        negative = [(f, v) for f, v in feature_impacts if v < 0]
        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        generate_advisor_report(
            student_id=f"STU-{actual_idx:05d}",
            risk_tier="High Risk",
            top_positive_features=positive[:5],
            top_negative_features=negative[:3],
        )


def run_shap_analysis() -> None:
    """Run full SHAP explainability pipeline."""
    model, X_test, y_test = load_model_and_data()
    shap_values = compute_shap_values(model, X_test)

    # Global summaries
    plot_summary_bar(shap_values, RESULTS_DIR / "shap_summary_bar.png")
    plot_beeswarm(shap_values, RESULTS_DIR / "shap_beeswarm.png")

    # Individual waterfalls
    generate_waterfall_gallery(model, X_test, y_test, shap_values)

    logger.info("Phase 8 SHAP Explainability complete!")


if __name__ == "__main__":
    run_shap_analysis()
