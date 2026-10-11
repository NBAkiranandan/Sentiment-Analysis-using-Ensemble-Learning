from pydantic import BaseModel, Field
from typing import Optional

class TrainingRequest(BaseModel):
    sample_size: Optional[int] = Field(None, description="Number of dataset records to train on (e.g., 50000 or 100000)")

class TrainingStatusResponse(BaseModel):
    status: str  # idle, training, completed, failed
    step: str
    progress: int  # 0 to 100
    message: str
    error: Optional[str] = None
