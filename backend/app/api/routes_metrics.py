from fastapi import APIRouter, HTTPException
from app.config import settings
from app.utils.file_utils import load_json
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["Metrics"])

@router.get("/models")
def get_available_models():
    """
    Get list of available machine learning models and active selection.
    """
    metadata_path = settings.MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        return {
            "models": ["Logistic Regression", "Random Forest", "AdaBoost"],
            "selected_model": None,
            "status": "not_trained",
            "message": "Models are not trained yet. Please train the models first."
        }

    try:
        meta = load_json(metadata_path)
        return {
            "models": ["Logistic Regression", "Random Forest", "AdaBoost"],
            "selected_model": meta.get("selected_model", "Logistic Regression"),
            "primary_metric": meta.get("primary_metric", "f1_score"),
            "status": "trained"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

@router.get("/metrics")
def get_model_metrics():
    """
    Get evaluation metrics for trained models.
    """
    metadata_path = settings.MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Models are not trained yet. Please train the models first.")

    try:
        meta = load_json(metadata_path)
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@router.get("/model-comparison")
def get_model_comparison():
    """
    Get side-by-side comparison among Logistic Regression (Baseline), Random Forest (Bagging), and AdaBoost (Boosting).
    """
    metadata_path = settings.MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Models are not trained yet. Please train the models first.")

    try:
        meta = load_json(metadata_path)
        lr = meta.get("logistic_regression", {})
        rf = meta.get("random_forest", {})
        ada = meta.get("adaboost", {})
        
        comparison = {
            "dataset_stats": meta.get("dataset_stats", {}),
            "models": ["Logistic Regression", "Random Forest", "AdaBoost"],
            "selected_model": meta.get("selected_model", "Logistic Regression"),
            "primary_metric": meta.get("primary_metric", "f1_score"),
            "metrics_comparison": {
                "accuracy": {
                    "logistic_regression": lr.get("accuracy"),
                    "random_forest": rf.get("accuracy"),
                    "adaboost": ada.get("accuracy")
                },
                "precision": {
                    "logistic_regression": lr.get("precision"),
                    "random_forest": rf.get("precision"),
                    "adaboost": ada.get("precision")
                },
                "recall": {
                    "logistic_regression": lr.get("recall"),
                    "random_forest": rf.get("recall"),
                    "adaboost": ada.get("recall")
                },
                "f1_score": {
                    "logistic_regression": lr.get("f1_score"),
                    "random_forest": rf.get("f1_score"),
                    "adaboost": ada.get("f1_score")
                },
                "training_time": {
                    "logistic_regression": lr.get("training_time"),
                    "random_forest": rf.get("training_time"),
                    "adaboost": ada.get("training_time")
                },
                "prediction_time": {
                    "logistic_regression": lr.get("prediction_time"),
                    "random_forest": rf.get("prediction_time"),
                    "adaboost": ada.get("prediction_time")
                }
            },
            "logistic_regression": lr,
            "random_forest": rf,
            "adaboost": ada
        }
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model comparison: {str(e)}")

@router.get("/methodology")
def get_methodology_info():
    """
    Get detailed methodology documentation for ML pipeline & ensemble approaches.
    """
    return {
        "title": "Sentiment Analysis and Model Comparison Methodology",
        "steps": [
            {
                "step": 1,
                "name": "Dataset Ingestion & Standardization",
                "description": "Utilizes the real Sentiment140 dataset (~1.6M tweets). Original target labels (0=Negative, 4=Positive) are standardized to 0 (Negative) and 1 (Positive)."
            },
            {
                "step": 2,
                "name": "Text Cleaning & Normalization",
                "description": "Applies Regex cleaning for URLs, @mentions, hashtags, punctuation, excessive whitespace, and lowercasing while strictly preserving sentiment-bearing words and negations (e.g., 'not', 'no', 'bad', 'great')."
            },
            {
                "step": 3,
                "name": "Tokenization & Lemmatization",
                "description": "Tokenizes cleaned tweets, filters stop words (excluding negations), and applies WordNet lemmatization to produce normalized text tokens."
            },
            {
                "step": 4,
                "name": "TF-IDF Vectorization & Anti-Data Leakage",
                "description": "Extracts unigram & bigram TF-IDF features (up to 25,000 features). Fitted strictly on training data ONLY (fit_transform on X_train, transform on X_test) to avoid data leakage."
            },
            {
                "step": 5,
                "name": "Model 1: Logistic Regression (Baseline)",
                "description": "Trains LogisticRegression(max_iter=1000) as a strong traditional Machine Learning baseline for linear text classification."
            },
            {
                "step": 6,
                "name": "Model 2: Random Forest (Bagging Ensemble)",
                "description": "Trains RandomForestClassifier(n_estimators=100) representing Bootstrap Aggregating (Bagging) to reduce variance."
            },
            {
                "step": 7,
                "name": "Model 3: AdaBoost (Boosting Ensemble)",
                "description": "Trains AdaBoostClassifier(n_estimators=100) representing Adaptive Boosting to reduce bias through sequential weak learners."
            },
            {
                "step": 8,
                "name": "Evaluation & Model Selection",
                "description": "Evaluates all 3 models on the identical 80/20 stratified test split. Calculates Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Execution Timings. Dynamically selects the final production model based on F1 Score."
            }
        ]
    }
