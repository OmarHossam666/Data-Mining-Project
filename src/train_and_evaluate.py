"""
Phase 6 & 7 — Model Training and Evaluation.

Trains an ensemble of classifiers on each checkpoint's feature-engineered
dataset and evaluates them on held-out test data.  Produces:
  - Trained model artifacts (``models/*.joblib``)
  - Metrics table (``results/metrics_table.csv``)
  - Checkpoint comparison (``results/checkpoint_comparison.csv``)
  - ROC curves, confusion matrices, checkpoint F1 comparison chart

Models:
  - Logistic Regression (baseline)
  - Random Forest (bagging ensemble)
  - XGBoost (gradient boosting ensemble)

Usage:
    python src/train_and_evaluate.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier

# Suppress XGBoost info logging
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Ensure src/ is importable
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _get_models() -> dict[str, object]:
    """Return dict of named model instances with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=config.RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=config.RANDOM_SEED,
            use_label_encoder=False,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_checkpoint_data(checkpoint_name: str) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Load train/val/test splits for a checkpoint."""
    features_dir = config.FEATURES_DIR

    train_df = pd.read_csv(features_dir / f"{checkpoint_name}_train.csv")
    val_df = pd.read_csv(features_dir / f"{checkpoint_name}_val.csv")
    test_df = pd.read_csv(features_dir / f"{checkpoint_name}_test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"].astype(int)
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"].astype(int)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    """Compute all evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Macro": f1_score(y_true, y_pred, average="macro"),
        "Kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = float("nan")
    else:
        metrics["ROC-AUC"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    results: list[dict], save_path: Path
) -> None:
    """Plot a grid of confusion matrices for the best checkpoint."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low Risk", "High Risk"],
                    yticklabels=["Low Risk", "High Risk"])
        ax.set_title(f"{r['model_name']}\nF1={r['metrics']['F1-Macro']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrices → %s", save_path)


def plot_roc_curves(results: list[dict], save_path: Path) -> None:
    """Plot ROC curves for all models on the best checkpoint."""
    plt.figure(figsize=(8, 6))

    for r in results:
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            auc = r["metrics"]["ROC-AUC"]
            plt.plot(fpr, tpr, label=f"{r['model_name']} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Checkpoint 12 (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curves → %s", save_path)


def plot_checkpoint_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    """Plot F1-Macro evolution across checkpoints for each model."""
    plt.figure(figsize=(10, 6))

    markers = {"Logistic Regression": "o", "Random Forest": "s", "XGBoost": "D"}
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}

    for model_name in comparison_df["Model"].unique():
        model_data = comparison_df[comparison_df["Model"] == model_name]
        plt.plot(
            model_data["Checkpoint"], model_data["Test_F1_Macro"],
            marker=markers.get(model_name, "o"),
            color=colors.get(model_name, "#333"),
            linewidth=2.5, markersize=10, label=model_name,
        )

    plt.xlabel("Semester Checkpoint", fontsize=12)
    plt.ylabel("F1-Macro Score (Test Set)", fontsize=12)
    plt.title("Early-Warning Prediction Accuracy Over Time", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved checkpoint comparison → %s", save_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training_and_evaluation() -> None:
    """Train models on all checkpoints and produce evaluation artifacts."""
    all_metrics: list[dict] = []
    comparison_rows: list[dict] = []
    best_checkpoint_results: list[dict] = []   # Results for checkpoint_12 visualizations

    for week in config.CHECKPOINT_WEEKS:
        cp_name = f"checkpoint_{week}"
        logger.info("=" * 60)
        logger.info("TRAINING: %s", cp_name)
        logger.info("=" * 60)

        X_train, y_train, X_val, y_val, X_test, y_test = load_checkpoint_data(cp_name)
        logger.info("Data shapes — train: %s, val: %s, test: %s",
                     X_train.shape, X_val.shape, X_test.shape)

        models = _get_models()

        for model_name, model in models.items():
            logger.info("Training %s on %s ...", model_name, cp_name)

            # Train
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_prob)
            logger.info("[%s | %s] F1=%.4f  AUC=%.4f  Kappa=%.4f",
                        cp_name, model_name,
                        metrics["F1-Macro"], metrics["ROC-AUC"], metrics["Kappa"])

            # Validation F1 for comparison
            y_val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, average="macro")

            # Store results
            all_metrics.append({
                "Checkpoint": f"Week {week}",
                "Model": model_name,
                **metrics,
            })
            comparison_rows.append({
                "Checkpoint": f"Week {week}",
                "Model": model_name,
                "Val_F1_Macro": val_f1,
                "Test_F1_Macro": metrics["F1-Macro"],
            })

            # Save model artifact for checkpoint 12 (best model)
            if week == 12:
                model_filename = {
                    "Logistic Regression": "baseline_lr.joblib",
                    "Random Forest": "rf_model.joblib",
                    "XGBoost": "xgb_model.joblib",
                }[model_name]
                joblib.dump(model, config.MODELS_DIR / model_filename)
                logger.info("Saved model → %s", model_filename)

                best_checkpoint_results.append({
                    "model_name": model_name,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "metrics": metrics,
                })

    # --- Save metrics tables ---
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(config.RESULTS_DIR / "metrics_table.csv", index=False)
    logger.info("Saved metrics_table.csv")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(config.RESULTS_DIR / "checkpoint_comparison.csv", index=False)
    logger.info("Saved checkpoint_comparison.csv")

    # --- Generate LaTeX table for paper ---
    cp12 = metrics_df[metrics_df["Checkpoint"] == "Week 12"].copy()
    cp12_tex = cp12[["Model", "Accuracy", "Precision", "Recall", "F1-Macro", "ROC-AUC", "Kappa"]]
    cp12_tex.to_latex(
        config.RESULTS_DIR / "metrics_table.tex",
        index=False, float_format="%.4f",
        caption="Model performance comparison at Checkpoint 12 (Week 12, test set).",
        label="tab:metrics",
    )
    logger.info("Saved metrics_table.tex")

    # --- Generate visualizations ---
    if best_checkpoint_results:
        plot_confusion_matrices(
            best_checkpoint_results,
            config.RESULTS_DIR / "confusion_matrices.png"
        )
        plot_roc_curves(
            best_checkpoint_results,
            config.RESULTS_DIR / "roc_curves.png"
        )

    plot_checkpoint_comparison(
        comparison_df,
        config.RESULTS_DIR / "figure1_checkpoint_comparison.png"
    )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY (Checkpoint 12 — Test Set)")
    logger.info("=" * 60)
    for r in best_checkpoint_results:
        m = r["metrics"]
        logger.info("  %-22s  F1=%.4f  AUC=%.4f  Kappa=%.4f",
                     r["model_name"], m["F1-Macro"], m["ROC-AUC"], m["Kappa"])


if __name__ == "__main__":
    run_training_and_evaluation()
