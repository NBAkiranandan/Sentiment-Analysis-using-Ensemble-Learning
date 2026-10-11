import pandas as pd
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.utils.file_utils import load_json
from app.utils.logger import logger

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

@router.get("/stats")
def get_dataset_stats():
    """
    Get dataset statistics, distribution, and sample records.
    """
    metadata_path = settings.MODEL_DIR / "model_metadata.json"
    processed_path = settings.PROCESSED_DATA_DIR / "processed_sentiment140.parquet"
    raw_path = settings.DATASET_PATH

    # Default fallback (shown only if metadata not yet generated)
    stats = {
        "records_raw": 1600000,
        "total_samples": 1600000,
        "positive_samples": 800000,
        "negative_samples": 800000,
        "is_sampled": False,
        "sample_size": 1600000,
        "training_mode": "Full Dataset",
        "samples": []
    }

    if metadata_path.exists():
        try:
            meta = load_json(metadata_path)
            ds = meta.get("dataset_stats", {})
            stats.update(ds)
            # Compute training_mode label
            is_sampled = ds.get("is_sampled", False)
            sample_size_used = ds.get("sample_size_used", ds.get("total_samples", 0))
            stats["training_mode"] = f"Sample ({sample_size_used:,} tweets)" if is_sampled else "Full Dataset"
        except Exception as e:
            logger.warning(f"Could not load metadata for dataset stats: {e}")

    # Load sample records from dataset
    try:
        if raw_path.exists():
            df_sample = pd.read_csv(raw_path, nrows=10, encoding="latin1", header=None)
            if df_sample.shape[1] == 6:
                df_sample.columns = ["target", "ids", "date", "flag", "user", "text"]
                samples = []
                for _, row in df_sample.iterrows():
                    samples.append({
                        "id": str(row["ids"]),
                        "user": str(row["user"]),
                        "text": str(row["text"]),
                        "target": int(row["target"]),
                        "sentiment": "Positive" if row["target"] in [4, 1, "4", "1"] else "Negative"
                    })
                stats["samples"] = samples
    except Exception as e:
        logger.warning(f"Could not load sample records: {e}")

    return stats
