import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import settings
from app.ml.preprocessing import TextPreprocessor
from app.ml.feature_engineering import TFIDFFeatureExtractor
from app.ml.logistic_regression_model import LogisticRegressionSentimentModel
from app.ml.random_forest_model import RandomForestSentimentModel
from app.ml.adaboost_model import AdaBoostSentimentModel
from app.utils.file_utils import load_json
from app.utils.logger import logger

class PredictionEngine:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFFeatureExtractor()
        self.lr_model = LogisticRegressionSentimentModel()
        self.rf_model = RandomForestSentimentModel()
        self.adaboost_model = AdaBoostSentimentModel()
        self.metadata: Optional[Dict[str, Any]] = None
        self.selected_model_name: str = "Logistic Regression"
        self.is_loaded: bool = False

    def load_artifacts(self) -> bool:
        """
        Load persisted vectorizer, models, and metadata.
        Returns True if loaded successfully, False if missing.
        """
        vectorizer_path = settings.MODEL_DIR / "tfidf_vectorizer.joblib"
        lr_path = settings.MODEL_DIR / "logistic_regression.joblib"
        rf_path = settings.MODEL_DIR / "random_forest.joblib"
        adaboost_path = settings.MODEL_DIR / "adaboost.joblib"
        metadata_path = settings.MODEL_DIR / "model_metadata.json"

        if not (vectorizer_path.exists() and lr_path.exists() and rf_path.exists() and adaboost_path.exists() and metadata_path.exists()):
            self.is_loaded = False
            return False

        try:
            self.vectorizer.load(vectorizer_path)
            self.lr_model.load(lr_path)
            self.rf_model.load(rf_path)
            self.adaboost_model.load(adaboost_path)
            self.metadata = load_json(metadata_path)
            self.selected_model_name = self.metadata.get("selected_model", "Logistic Regression")
            self.is_loaded = True
            logger.info(f"Prediction engine loaded successfully. Active selected model: {self.selected_model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            self.is_loaded = False
            return False

    def _get_model_by_name(self, name: str):
        normalized = name.lower().replace("_", " ").replace("-", " ").strip()
        if "logistic" in normalized or "baseline" in normalized:
            return self.lr_model, "Logistic Regression"
        elif "adaboost" in normalized or "boosting" in normalized:
            return self.adaboost_model, "AdaBoost"
        elif "forest" in normalized or "bagging" in normalized:
            return self.rf_model, "Random Forest"
        else:
            # Fallback to selected model
            return self._get_model_by_name(self.selected_model_name)

    def predict_single(self, text: str, model_choice: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict sentiment for a single input text using chosen or selected model.
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError("Models are not trained yet. Please train the models first.")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        # Text preprocessing
        processed = self.preprocessor.tokenize_and_filter(text)
        if not processed:
            processed = self.preprocessor.clean_text(text)

        # Vectorization
        X = self.vectorizer.transform([processed])

        # Select model
        model, chosen_model_name = self._get_model_by_name(model_choice or self.selected_model_name)

        # Predict
        pred_label = int(model.predict(X)[0])
        probas = model.predict_proba(X)[0]
        confidence = float(probas[pred_label])
        sentiment = "Positive" if pred_label == 1 else "Negative"

        return {
            "text": text,
            "processed_text": processed,
            "sentiment": sentiment,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "model": chosen_model_name,
            "probabilities": {
                "Negative": round(float(probas[0]), 4),
                "Positive": round(float(probas[1]), 4)
            }
        }

    def predict_compare(self, text: str) -> Dict[str, Any]:
        """
        Evaluate single input text across ALL 3 models (Logistic Regression, Random Forest, AdaBoost).
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError("Models are not trained yet. Please train the models first.")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        processed = self.preprocessor.tokenize_and_filter(text)
        if not processed:
            processed = self.preprocessor.clean_text(text)

        X = self.vectorizer.transform([processed])

        models_dict = {
            "logistic_regression": (self.lr_model, "Logistic Regression", "Baseline"),
            "random_forest": (self.rf_model, "Random Forest", "Bagging"),
            "adaboost": (self.adaboost_model, "AdaBoost", "Boosting")
        }

        results = {}
        for key, (model, display_name, m_type) in models_dict.items():
            label = int(model.predict(X)[0])
            probas = model.predict_proba(X)[0]
            conf = float(probas[label])
            results[key] = {
                "name": display_name,
                "type": m_type,
                "sentiment": "Positive" if label == 1 else "Negative",
                "label": label,
                "confidence": round(conf, 4),
                "probabilities": {
                    "Negative": round(float(probas[0]), 4),
                    "Positive": round(float(probas[1]), 4)
                }
            }

        return {
            "text": text,
            "processed_text": processed,
            "selected_model": self.selected_model_name,
            "logistic_regression": results["logistic_regression"],
            "random_forest": results["random_forest"],
            "adaboost": results["adaboost"]
        }

    def predict_batch(self, df: pd.DataFrame, text_column: str, model_choice: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a DataFrame of texts.
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError("Models are not trained yet. Please train the models first.")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in uploaded CSV.")

        texts = df[text_column].astype(str).tolist()
        processed_texts = self.preprocessor.transform_series(texts)
        processed_texts = [p if p else "neutral" for p in processed_texts]

        X = self.vectorizer.transform(processed_texts)

        model, chosen_model_name = self._get_model_by_name(model_choice or self.selected_model_name)

        preds = model.predict(X)
        probas = model.predict_proba(X)

        results = []
        for i in range(len(texts)):
            label = int(preds[i])
            conf = float(probas[i][label])
            results.append({
                "original_text": texts[i],
                "processed_text": processed_texts[i],
                "predicted_sentiment": "Positive" if label == 1 else "Negative",
                "label": label,
                "confidence": round(conf, 4),
                "model": chosen_model_name
            })
        return results

prediction_engine = PredictionEngine()

