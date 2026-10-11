from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class ModelMetric(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    training_time: float
    prediction_time: float

class DatasetStats(BaseModel):
    dataset_name: Optional[str] = "Sentiment140"
    total_samples: int
    positive_samples: int
    negative_samples: int
    train_size: int
    test_size: int
    is_sampled: bool
    sample_size: int
    sample_size_used: Optional[int] = None
    training_mode: Optional[str] = "Development Sample"

class ModelComparisonResponse(BaseModel):
    dataset_stats: DatasetStats
    models: List[str] = ["Logistic Regression", "Random Forest", "AdaBoost"]
    logistic_regression: ModelMetric
    random_forest: ModelMetric
    adaboost: ModelMetric
    selected_model: str
    primary_metric: str = "f1_score"
