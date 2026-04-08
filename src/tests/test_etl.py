import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src folder to path so we can import our ETL modules
sys.path.append(str(Path.cwd() / "src"))
from etl.extract import extract_data
from etl.transform import transform_data

# Use pytest fixtures to load data once for all tests
@pytest.fixture(scope="module")
def raw_data():
    return extract_data()

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
                assert cls in [0, 1], f"Unexpected class '{cls}' found in risk_label of {name}. Expected only 0 or 1."