import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Sentiment Analysis Ensemble API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Directories
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"

    # Default Dataset Path
    DATASET_PATH: Path = BASE_DIR / "data" / "raw" / "sentiment140.csv"

    # Training Parameters
    DATASET_SAMPLE_SIZE: int = 50000  # Configurable: e.g. 50000 or 100000 for high performance
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # TF-IDF Parameters
    TFIDF_MAX_FEATURES: int = 25000
    TFIDF_NGRAM_MIN: int = 1
    TFIDF_NGRAM_MAX: int = 2
    TFIDF_MIN_DF: int = 2
    TFIDF_MAX_DF: float = 0.95

    # Logistic Regression Parameters (Baseline)
    LOGISTIC_MAX_ITER: int = 1000
    LOGISTIC_C: float = 1.0

    # Random Forest Parameters
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 25
    RF_MIN_SAMPLES_SPLIT: int = 2
    RF_MIN_SAMPLES_LEAF: int = 1

    # AdaBoost Parameters
    ADABOOST_N_ESTIMATORS: int = 100
    ADABOOST_LEARNING_RATE: float = 1.0

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
