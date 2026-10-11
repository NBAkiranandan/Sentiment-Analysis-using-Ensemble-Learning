import os
import sys
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.logger import logger

STANFORD_SENTIMENT140_ZIP_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

def download_real_sentiment140(target_path: Path) -> Path:
    """
    Downloads and extracts the official, authentic Stanford Sentiment140 dataset.
    Contains ~1.6 million real annotated tweets.
    """
    logger.info(f"Downloading authentic Stanford Sentiment140 dataset from {STANFORD_SENTIMENT140_ZIP_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    response = requests.get(STANFORD_SENTIMENT140_ZIP_URL, stream=True, headers=headers, timeout=120)
    response.raise_for_status()

    logger.info("Download completed. Extracting training.1600000.processed.noemoticon.csv...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            if "training" in name and name.endswith(".csv"):
                logger.info(f"Extracting {name} to {target_path}...")
                with z.open(name) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                break

    logger.info(f"Successfully extracted real Sentiment140 dataset to {target_path} ({target_path.stat().st_size} bytes).")
    return target_path

def download_or_prepare_dataset(force_redownload: bool = False):
    target_path = settings.DATASET_PATH
    
    if target_path.exists() and not force_redownload:
        try:
            # Verify file size is substantial (real dataset is > 200MB)
            if target_path.stat().st_size > 50 * 1024 * 1024:
                df_sample = pd.read_csv(target_path, nrows=5, header=None, encoding="latin1")
                if df_sample.shape[1] == 6 and (str(df_sample.iloc[0, 0]).strip() in ["0", "4"]):
                    logger.info(f"Real Sentiment140 dataset verified at {target_path} ({target_path.stat().st_size} bytes).")
                    return
        except Exception as e:
            logger.warning(f"Existing dataset file check failed ({e}). Re-downloading real Sentiment140 dataset...")

    download_real_sentiment140(target_path)

if __name__ == "__main__":
    download_or_prepare_dataset(force_redownload=True)
