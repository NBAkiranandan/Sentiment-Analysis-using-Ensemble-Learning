import threading
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional

from app.schemas.training_schema import TrainingRequest, TrainingStatusResponse
from app.ml.ensemble_manager import EnsembleManager, training_tracker
from app.ml.prediction import prediction_engine
from app.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["Training"])
ensemble_manager = EnsembleManager()

def execute_training_task(sample_size: Optional[int] = None):
    try:
        ensemble_manager.run_full_training_pipeline(sample_size=sample_size)
        prediction_engine.load_artifacts()
    except Exception as e:
        logger.error(f"Background training failed: {str(e)}")

@router.post("/train", response_model=TrainingStatusResponse)
def start_training(request: Optional[TrainingRequest] = None, background_tasks: BackgroundTasks = None):
    """
    Trigger model training in the background.
    """
    if training_tracker.status == "training":
        return training_tracker.to_dict()

    sample_size = request.sample_size if (request and request.sample_size is not None) else settings.DATASET_SAMPLE_SIZE

    # Launch background thread
    thread = threading.Thread(target=execute_training_task, args=(sample_size,))
    thread.start()

    training_tracker.update("training", "started", 5, f"Training process initiated (sample_size={sample_size})...")
    return training_tracker.to_dict()

@router.get("/training/status", response_model=TrainingStatusResponse)
def get_training_status():
    """
    Get current model training status and progress.
    """
    return training_tracker.to_dict()
