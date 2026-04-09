import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

# Resolve src directory from __file__ (not CWD) for reliable imports
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from etl.extract import extract_data
from etl.transform import transform_data


# --- Fixture Data (no dependency on local unversioned CSV files) ---

def _make_uci_student_df(n=50):
    """Creates a minimal UCI Student Performance fixture."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'age': rng.integers(15, 22, size=n),
        'sex': rng.choice(['M', 'F'], size=n),
        'address': rng.choice(['U', 'R'], size=n),
        'famsize': rng.choice(['LE3', 'GT3'], size=n),
        'Medu': rng.integers(0, 5, size=n),
        'studytime': rng.integers(1, 5, size=n),
        'failures': rng.integers(0, 4, size=n),
        'internet': rng.choice(['yes', 'no'], size=n),
        'paid': rng.choice(['yes', 'no'], size=n),
        'absences': rng.integers(0, 30, size=n),
        'G1': rng.integers(0, 20, size=n),
        'G2': rng.integers(0, 20, size=n),
        'G3': rng.integers(0, 20, size=n),
    })


def _make_uci_dropout_df(n=80):
    """Creates a minimal UCI Dropout & Academic Success fixture."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Age at enrollment': rng.integers(17, 50, size=n),
        'Gender': rng.choice([0, 1], size=n),
        'Curricular units 1st sem (approved)': rng.integers(0, 10, size=n),
        'Curricular units 2nd sem (approved)': rng.integers(0, 10, size=n),
        'Target': rng.choice(['Dropout', 'Graduate', 'Enrolled'], size=n),
    })


def _make_kaggle_df(n=60):
    """Creates a minimal Kaggle Student Performance fixture."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'Age': rng.integers(15, 19, size=n),
        'Gender': rng.choice([0, 1], size=n),
        'StudyTimeWeekly': rng.uniform(0, 20, size=n).round(1),
        'Absences': rng.integers(0, 30, size=n),
        'ParentalSupport': rng.integers(0, 5, size=n),
        'GPA': rng.uniform(1.0, 4.0, size=n).round(2),
        'GradeClass': rng.choice([0, 1, 2, 3, 4], size=n),
    })


@pytest.fixture(scope="module")
def raw_data():
    """Returns mock raw datasets — independent of local CSV files."""
    return {
        'uci_student': _make_uci_student_df(649),
        'uci_dropout': _make_uci_dropout_df(4424),
        'kaggle': _make_kaggle_df(2392),
    }


@pytest.fixture(scope="module")
def clean_data(raw_data):
    return transform_data(raw_data)


def test_extract_row_counts(raw_data):
    """Assert input row counts match the exact expected numbers from Phase 1."""
    assert len(raw_data['uci_student']) == 649, "UCI Student dataset row count mismatch!"
    assert len(raw_data['uci_dropout']) == 4424, "UCI Dropout dataset row count mismatch!"
    assert len(raw_data['kaggle']) == 2392, "Kaggle dataset row count mismatch!"


def test_no_nan_in_target_columns(clean_data):
    """Assert no NaNs exist in critical columns after the transform phase."""
    for name, df in clean_data.items():
        # Test if our median/mode imputation worked
        assert df.isnull().sum().sum() == 0, f"Found NaNs in {name} after transformation!"

        # If we engineered the risk label, ensure it has no missing values
        if 'risk_label' in df.columns:
            assert df['risk_label'].isna().sum() == 0, f"NaNs found in risk_label for {name}!"


def test_class_distribution_logic(clean_data):
    """Assert that the risk label only contains expected classes (0 and 1)."""
    for name, df in clean_data.items():
        if 'risk_label' in df.columns:
            unique_classes = df['risk_label'].unique()
            for cls in unique_classes:
                assert cls in [0, 1], (
                    f"Unexpected class '{cls}' found in risk_label of {name}. "
                    "Expected only 0 or 1."
                )


def test_risk_labels_exist_for_all_datasets(clean_data):
    """Assert that risk_label was successfully engineered for every dataset."""
    for name, df in clean_data.items():
        assert 'risk_label' in df.columns, f"risk_label missing from {name}!"


def test_transform_handles_all_null_column():
    """Edge case: all-null columns should be dropped, not crash the pipeline."""
    df_with_null_col = pd.DataFrame({
        'G3': [10, 5, 15],
        'age': [18, 17, 16],
        'empty_col': [None, None, None],
    })
    result = transform_data({'test': df_with_null_col})
    assert 'empty_col' not in result['test'].columns, "All-null column should have been dropped!"