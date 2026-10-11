from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class SinglePredictionRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "I really enjoyed this product!"})
    model: Optional[str] = Field(None, json_schema_extra={"example": "Logistic Regression"})

class SinglePredictionResponse(BaseModel):
    text: str
    processed_text: str
    sentiment: str
    label: int
    confidence: float
    model: str
    probabilities: Dict[str, float]

class ModelPredictionDetail(BaseModel):
    name: str
    type: str
    sentiment: str
    label: int
    confidence: float
    probabilities: Dict[str, float]

class ComparePredictionResponse(BaseModel):
    text: str
    processed_text: str
    selected_model: str
    logistic_regression: ModelPredictionDetail
    random_forest: ModelPredictionDetail
    adaboost: ModelPredictionDetail

class BatchPredictionItem(BaseModel):
    original_text: str
    processed_text: str
    predicted_sentiment: str
    label: int
    confidence: float
    model: str

class BatchPredictionResponse(BaseModel):
    total_records: int
    positive_count: int
    negative_count: int
    model_used: str
    results: List[BatchPredictionItem]
