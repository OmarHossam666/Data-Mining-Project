import pandas as pd
import logging
from pathlib import Path

# Resolve project root from __file__ (not CWD) for portability
SRC_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(SRC_DIR))
import config

logger = logging.getLogger(__name__)

def extract_data():
    """Extracts raw datasets and returns them in a dictionary."""
    logger.info("Starting extraction phase...")

    datasets = {}

    try:
        datasets['uci_student'] = pd.read_csv(config.RAW_DATA_DIR / "uci_student_performance.csv")
        datasets['uci_dropout'] = pd.read_csv(config.RAW_DATA_DIR / "uci_dropout_success.csv")
        datasets['kaggle'] = pd.read_csv(config.RAW_DATA_DIR / "kaggle_student_performance.csv")

        for name, df in datasets.items():
            logger.info(f"Extracted {name} successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

        return datasets

    except FileNotFoundError as e:
        logger.error(f"Extraction failed. File not found: {e}")
        raise