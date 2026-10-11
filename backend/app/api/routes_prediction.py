import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from app.schemas.prediction_schema import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    ComparePredictionResponse,
    BatchPredictionResponse
)
from app.ml.prediction import prediction_engine
from app.utils.file_utils import detect_columns
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/predict", response_model=SinglePredictionResponse)
def predict_sentiment(request: SinglePredictionRequest):
    """
    Predict sentiment for a single text input using specified model or selected best model.
    """
    try:
        result = prediction_engine.predict_single(request.text, model_choice=request.model)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/compare", response_model=ComparePredictionResponse)
def predict_sentiment_compare(request: SinglePredictionRequest):
    """
    Evaluate single text input across ALL 3 models: Logistic Regression, Random Forest, AdaBoost.
    """
    try:
        result = prediction_engine.predict_compare(request.text)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Prediction compare error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction comparison failed: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_csv(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    text_column: Optional[str] = Form(None)
):
    """
    Predict sentiment for a batch of text samples uploaded via CSV file.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding="latin1")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        # Automatically detect text column if not specified
        if not text_column or text_column not in df.columns:
            detected_text, _ = detect_columns(df)
            text_column = detected_text

        results = prediction_engine.predict_batch(df, text_column=text_column, model_choice=model)

        pos_count = sum(1 for r in results if r["label"] == 1)
        neg_count = sum(1 for r in results if r["label"] == 0)
        model_name = results[0]["model"] if results else (model or "Selected Model")

        return {
            "total_records": len(results),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "model_used": model_name,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
