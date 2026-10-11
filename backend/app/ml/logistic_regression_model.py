import time
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression

from app.config import settings
from app.utils.file_utils import save_joblib, load_joblib
from app.utils.logger import logger

class LogisticRegressionSentimentModel:
    """
    Baseline machine learning model using Logistic Regression.
    """
    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = settings.RANDOM_STATE,
        solver: str = "lbfgs"
    ):
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.solver = solver

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            solver=self.solver,
            n_jobs=-1
        )
        self.is_trained = False
        self.training_time = 0.0

    def fit(self, X_train, y_train):
        logger.info(f"Training Logistic Regression (Baseline) model (max_iter={self.max_iter}, C={self.C})...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = round(time.time() - start_time, 4)
        self.is_trained = True
        logger.info(f"Logistic Regression trained successfully in {self.training_time} seconds.")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Logistic Regression model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Logistic Regression model is not trained yet.")
        return self.model.predict_proba(X)

    def save(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "logistic_regression.joblib"
        save_joblib(self.model, filepath)

    def load(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "logistic_regression.joblib"
        self.model = load_joblib(filepath)
        self.is_trained = True
        logger.info(f"Logistic Regression model loaded from {filepath}")
