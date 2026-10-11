from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Any
from pathlib import Path
from app.config import settings
from app.utils.file_utils import save_joblib, load_joblib
from app.utils.logger import logger

class TFIDFFeatureExtractor:
    def __init__(
        self,
        max_features: int = settings.TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = (settings.TFIDF_NGRAM_MIN, settings.TFIDF_NGRAM_MAX),
        min_df: int = settings.TFIDF_MIN_DF,
        max_df: float = settings.TFIDF_MAX_DF,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf
        )
        self.is_fitted = False

    def fit_transform(self, raw_documents):
        logger.info(f"Fitting TF-IDF Vectorizer (max_features={self.max_features}, ngrams={self.ngram_range})...")
        X = self.vectorizer.fit_transform(raw_documents)
        self.is_fitted = True
        logger.info(f"TF-IDF Vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}, Output shape: {X.shape}")
        return X

    def transform(self, raw_documents):
        if not self.is_fitted:
            raise ValueError("TF-IDF Vectorizer is not fitted yet.")
        return self.vectorizer.transform(raw_documents)

    def save(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "tfidf_vectorizer.joblib"
        save_joblib(self.vectorizer, filepath)

    def load(self, filepath: Path = None):
        if filepath is None:
            filepath = settings.MODEL_DIR / "tfidf_vectorizer.joblib"
        self.vectorizer = load_joblib(filepath)
        self.is_fitted = True
        logger.info(f"TF-IDF Vectorizer loaded successfully from {filepath}")
