import time
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path
from app.config import settings
from app.utils.file_utils import save_joblib, load_joblib
from app.utils.logger import logger

class AdaBoostSentimentModel:
    """
    Boosting-based ensemble model using AdaBoostClassifier.
    """
    def __init__(
        self,
        n_estimators: int = settings.ADABOOST_N_ESTIMATORS,
        learning_rate: float = settings.ADABOOST_LEARNING_RATE,
        random_state: int = settings.RANDOM_STATE
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.is_trained = False
        self.training_time = 0.0

    def fit(self, X_train, y_train):
        logger.info(f"Training AdaBoost (Boosting) model (n_estimators={self.n_estimators}, learning_rate={self.learning_rate})...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = round(time.time() - start_time, 4)
        self.is_trained = True
        logger.info(f"AdaBoost trained successfully in {self.training_time} seconds.")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("AdaBoost model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("AdaBoost model is not trained yet.")
        return self.model.predict_proba(X)

    def save(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "adaboost.joblib"
        save_joblib(self.model, filepath)

    def load(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "adaboost.joblib"
        self.model = load_joblib(filepath)
        self.is_trained = True
        logger.info(f"AdaBoost model loaded from {filepath}")
