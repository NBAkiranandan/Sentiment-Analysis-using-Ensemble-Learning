import time
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import Dict, Any
from app.config import settings
from app.utils.file_utils import save_joblib, load_joblib
from app.utils.logger import logger

class RandomForestSentimentModel:
    """
    Bagging-based ensemble model using RandomForestClassifier.
    """
    def __init__(
        self,
        n_estimators: int = settings.RF_N_ESTIMATORS,
        max_depth: int = settings.RF_MAX_DEPTH,
        min_samples_split: int = settings.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = settings.RF_MIN_SAMPLES_LEAF,
        random_state: int = settings.RANDOM_STATE,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.is_trained = False
        self.training_time = 0.0

    def fit(self, X_train, y_train):
        logger.info(f"Training Random Forest (Bagging) model (n_estimators={self.n_estimators}, max_depth={self.max_depth})...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = round(time.time() - start_time, 4)
        self.is_trained = True
        logger.info(f"Random Forest trained successfully in {self.training_time} seconds.")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Random Forest model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Random Forest model is not trained yet.")
        return self.model.predict_proba(X)

    def save(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "random_forest.joblib"
        save_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "random_forest.joblib"
        self.model = load_joblib(filepath)
        self.is_trained = True
        logger.info(f"Random Forest model loaded from {filepath}")
