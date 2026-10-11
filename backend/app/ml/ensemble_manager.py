import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split

from app.config import settings
from app.ml.preprocessing import TextPreprocessor
from app.ml.feature_engineering import TFIDFFeatureExtractor
from app.ml.logistic_regression_model import LogisticRegressionSentimentModel
from app.ml.random_forest_model import RandomForestSentimentModel
from app.ml.adaboost_model import AdaBoostSentimentModel
from app.ml.evaluation import evaluate_model_performance
from app.utils.file_utils import save_json, load_json, load_and_standardize_dataset
from app.utils.logger import logger

class TrainingStatusTracker:
    def __init__(self):
        self.status = "idle"  # idle, training, completed, failed
        self.step = "none"    # dataset_loading, preprocessing, tfidf, lr_training, rf_training, adaboost_training, evaluation, saving
        self.progress = 0     # 0 to 100
        self.message = "System ready."
        self.error = None

    def update(self, status: str, step: str, progress: int, message: str, error: str = None):
        self.status = status
        self.step = step
        self.progress = progress
        self.message = message
        self.error = error
        logger.info(f"Training Status Update [{progress}%] ({status}/{step}): {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "error": self.error
        }

training_tracker = TrainingStatusTracker()

class EnsembleManager:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFFeatureExtractor()
        self.lr_model = LogisticRegressionSentimentModel()
        self.rf_model = RandomForestSentimentModel()
        self.adaboost_model = AdaBoostSentimentModel()
        self.metadata_path = settings.MODEL_DIR / "model_metadata.json"

    def run_full_training_pipeline(
        self,
        dataset_path: Optional[Path] = None,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete training, evaluation, comparison, and persistence pipeline.
        Trains and compares 3 models: Logistic Regression (Baseline), Random Forest (Bagging), AdaBoost (Boosting).
        """
        if dataset_path is None:
            dataset_path = settings.DATASET_PATH
        if sample_size is None:
            sample_size = settings.DATASET_SAMPLE_SIZE

        try:
            # 1. Dataset Loading
            training_tracker.update("training", "dataset_loading", 10, f"Loading dataset from {dataset_path}...")
            df = load_and_standardize_dataset(dataset_path, sample_size=sample_size)
            
            total_samples = len(df)
            pos_samples = int((df["target"] == 1).sum())
            neg_samples = int((df["target"] == 0).sum())

            # 2. Text Preprocessing
            training_tracker.update("training", "preprocessing", 25, f"Preprocessing text data ({total_samples} samples)...")
            processed_texts = self.preprocessor.transform_series(df["text"])
            df["processed_text"] = processed_texts
            
            # Save processed dataset copy for stats/inspection
            processed_data_file = settings.PROCESSED_DATA_DIR / "processed_sentiment140.parquet"
            try:
                df.to_parquet(processed_data_file, index=False)
            except Exception:
                df.to_csv(settings.PROCESSED_DATA_DIR / "processed_sentiment140.csv", index=False)

            # 3. Train/Test Split (80/20 Stratified)
            training_tracker.update("training", "tfidf", 40, "Splitting dataset (80% train, 20% test) and extracting TF-IDF features...")
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df["processed_text"],
                df["target"],
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE,
                stratify=df["target"]
            )

            # 4. TF-IDF Fit Transform on TRAIN ONLY
            X_train = self.vectorizer.fit_transform(X_train_text)
            X_test = self.vectorizer.transform(X_test_text)
            self.vectorizer.save()

            # 5. Train Model 1: Logistic Regression (Baseline)
            training_tracker.update("training", "lr_training", 50, "Training Logistic Regression classifier (Baseline)...")
            self.lr_model.fit(X_train, y_train)
            self.lr_model.save()

            # 6. Train Model 2: Random Forest (Bagging)
            training_tracker.update("training", "rf_training", 70, "Training Random Forest classifier (Bagging)...")
            self.rf_model.fit(X_train, y_train)
            self.rf_model.save()

            # 7. Train Model 3: AdaBoost (Boosting)
            training_tracker.update("training", "adaboost_training", 85, "Training AdaBoost classifier (Boosting)...")
            self.adaboost_model.fit(X_train, y_train)
            self.adaboost_model.save()

            # 8. Evaluate All Three Models
            training_tracker.update("training", "evaluation", 95, "Evaluating performance for all 3 models...")
            lr_metrics = evaluate_model_performance(self.lr_model, X_test, y_test, "Logistic Regression")
            rf_metrics = evaluate_model_performance(self.rf_model, X_test, y_test, "Random Forest")
            adaboost_metrics = evaluate_model_performance(self.adaboost_model, X_test, y_test, "AdaBoost")

            # 9. Dynamic Model Selection based on F1 Score
            candidates = [
                ("Logistic Regression", lr_metrics["f1_score"]),
                ("Random Forest", rf_metrics["f1_score"]),
                ("AdaBoost", adaboost_metrics["f1_score"])
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_model_name, best_f1 = candidates[0]

            metadata = {
                "dataset_stats": {
                    "dataset_name": "Sentiment140",
                    "records_raw": 1600000,
                    "total_samples": total_samples,
                    "positive_samples": pos_samples,
                    "negative_samples": neg_samples,
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "is_sampled": total_samples < 1600000,
                    "sample_size": total_samples,
                    "sample_size_used": total_samples,
                    "training_mode": f"Development Sample ({total_samples:,} tweets)" if total_samples < 1600000 else "Full Dataset",
                    "dataset_source": str(dataset_path.name)
                },
                "parameters": {
                    "tfidf_max_features": self.vectorizer.max_features,
                    "tfidf_ngram_range": list(self.vectorizer.ngram_range),
                    "test_size": settings.TEST_SIZE,
                    "random_state": settings.RANDOM_STATE,
                    "logistic_max_iter": self.lr_model.max_iter,
                    "rf_n_estimators": self.rf_model.n_estimators,
                    "adaboost_n_estimators": self.adaboost_model.n_estimators
                },
                "models": ["Logistic Regression", "Random Forest", "AdaBoost"],
                "logistic_regression": lr_metrics,
                "random_forest": rf_metrics,
                "adaboost": adaboost_metrics,
                "selected_model": best_model_name,
                "primary_metric": "f1_score"
            }

            save_json(metadata, self.metadata_path)
            training_tracker.update("completed", "saving", 100, f"Training complete! Best model selected: {best_model_name} (F1: {best_f1})")
            return metadata

        except Exception as e:
            logger.error(f"Error during training pipeline: {str(e)}", exc_info=True)
            training_tracker.update("failed", "error", 0, f"Training failed: {str(e)}", error=str(e))
            raise e
