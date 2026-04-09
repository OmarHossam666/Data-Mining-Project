import logging
from pathlib import Path

# --- Path Configuration ---
# This points to the root of your project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define standard data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Database Configuration ---
# SQLite for local dev (can be changed to PostgreSQL string later)
DB_PATH = PROJECT_ROOT / "warehouse.db"
DATABASE_URI = f"sqlite:///{DB_PATH}"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Project configuration loaded and directories verified.")