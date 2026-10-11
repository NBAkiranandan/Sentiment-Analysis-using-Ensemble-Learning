import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ml.preprocessing import TextPreprocessor
from app.utils.file_utils import load_and_standardize_dataset
from app.utils.logger import logger

def preprocess_dataset_cli():
    logger.info("Starting dataset preprocessing pipeline CLI...")
    raw_path = settings.DATASET_PATH
    
    if not raw_path.exists():
        logger.info("Dataset file missing. Running download script first...")
        from scripts.download_dataset import download_or_prepare_dataset
        download_or_prepare_dataset()

    df = load_and_standardize_dataset(raw_path, sample_size=settings.DATASET_SAMPLE_SIZE)
    
    preprocessor = TextPreprocessor()
    logger.info("Cleaning and tokenizing text data...")
    df["processed_text"] = preprocessor.transform_series(df["text"])

    output_path = settings.PROCESSED_DATA_DIR / "processed_sentiment140.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessing completed. Processed dataset saved to {output_path} ({len(df)} records).")

if __name__ == "__main__":
    preprocess_dataset_cli()
