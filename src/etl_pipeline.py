import logging
import sys
from pathlib import Path

# Setup logging to output to a file and the console
log_path = Path(__file__).resolve().parent.parent / "etl_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import ETL modules
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data

def run_pipeline():
    logger.info("=== Starting ProGrade ETL Pipeline ===")
    try:
        # Extract
        raw_data = extract_data()
        
        # Transform
        clean_data = transform_data(raw_data)
        
        # Load
        load_data(clean_data)
        
        logger.info("=== ETL Pipeline Completed Successfully ===")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()