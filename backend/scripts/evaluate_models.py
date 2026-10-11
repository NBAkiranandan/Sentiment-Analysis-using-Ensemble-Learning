import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.file_utils import load_json
from app.utils.logger import logger

def evaluate_cli():
    metadata_path = settings.MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}. Please run train_models.py first.")
        sys.exit(1)

    metadata = load_json(metadata_path)
    rf = metadata["random_forest"]
    ada = metadata["adaboost"]

    print("\n==================================================")
    print("           MODEL PERFORMANCE COMPARISON           ")
    print("==================================================")
    print(f"{'Metric':<18} | {'Random Forest':<15} | {'AdaBoost':<15}")
    print("--------------------------------------------------")
    print(f"{'Accuracy':<18} | {rf['accuracy']:<15.4f} | {ada['accuracy']:<15.4f}")
    print(f"{'Precision':<18} | {rf['precision']:<15.4f} | {ada['precision']:<15.4f}")
    print(f"{'Recall':<18} | {rf['recall']:<15.4f} | {ada['recall']:<15.4f}")
    print(f"{'F1-Score':<18} | {rf['f1_score']:<15.4f} | {ada['f1_score']:<15.4f}")
    print(f"{'Training Time (s)':<18} | {rf['training_time']:<15.2f} | {ada['training_time']:<15.2f}")
    print(f"{'Predict Time (s)':<18} | {rf['prediction_time']:<15.4f} | {ada['prediction_time']:<15.4f}")
    print("--------------------------------------------------")
    print(f"Selected Model: {metadata['selected_model']}")
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_cli()
