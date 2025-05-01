import os
import logging
from pathlib import Path

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define folders and patterns to clean
CLEANUP_CONFIG = [
    {"folder": "./logs", "patterns": ["*.log"]},  # Logs
    {"folder": "./assets/figures", "patterns": ["*.png"]},  # Figures
    {"folder": "./data/raw_data", "patterns": ["~lock.*"]},  # Temp Files
    {"folder": "./__pycache__", "patterns": ["*.pyc", "*.pyo"]},  # Python cache
    {"folder": "./weights", "patterns": ["*.pkl"]},
    {"folder": "./reports", "patterns": ["*.png", "*.log","*.txt"]},
    {"folder": "./reports/figures", "patterns": ["*.png"]},
]

def clean_folder(folder_path: str, patterns: list) -> None:
    """
    Cleans a folder by removing files matching specified patterns.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"Folder does not exist: {folder}")
        return

    for pattern in patterns:
        for file in folder.glob(pattern):
            try:
                if file.is_file():
                    file.unlink()  # Delete the file
                    logger.info(f"Deleted file: {file}")
                elif file.is_dir():
                    # Remove the directory recursively
                    os.rmdir(file)
                    logger.info(f"Deleted folder: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")

def main():
    """
    Goes through the CLEANUP_CONFIG list and cleans up files in specified folders.
    """
    for config in CLEANUP_CONFIG:
        folder = config.get("folder")
        patterns = config.get("patterns", [])
        logger.info(f"Cleaning folder: {folder} with patterns: {patterns}")
        clean_folder(folder, patterns)
    logger.info("Project cleanup completed.")

if __name__ == "__main__":
    main()