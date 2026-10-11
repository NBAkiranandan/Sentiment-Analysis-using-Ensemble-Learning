import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ml.ensemble_manager import EnsembleManager
from scripts.download_dataset import download_or_prepare_dataset
from app.utils.logger import logger

def train_cli():
    logger.info("Starting model training CLI...")
    if not settings.DATASET_PATH.exists():
        logger.info("Dataset not found. Downloading/preparing dataset...")
        download_or_prepare_dataset()

    manager = EnsembleManager()
    metadata = manager.run_full_training_pipeline(sample_size=settings.DATASET_SAMPLE_SIZE)
    
    lr = metadata["logistic_regression"]
    rf = metadata["random_forest"]
    ada = metadata["adaboost"]
    
    print("\n==================================================")
    print("      TRAINING & MODEL COMPARISON SUMMARY          ")
    print("==================================================")
    print(f"Dataset Size Evaluated: {metadata['dataset_stats']['total_samples']:,}")
    print(f"Train/Test Split:       80% / 20% (Stratified)")
    print("--------------------------------------------------")
    print("LOGISTIC REGRESSION (BASELINE):")
    print(f"  Accuracy:  {lr['accuracy']:.4f}")
    print(f"  Precision: {lr['precision']:.4f}")
    print(f"  Recall:    {lr['recall']:.4f}")
    print(f"  F1-Score:  {lr['f1_score']:.4f}")
    print(f"  Time:      {lr['training_time']:.2f}s")
    print("--------------------------------------------------")
    print("RANDOM FOREST (BAGGING):")
    print(f"  Accuracy:  {rf['accuracy']:.4f}")
    print(f"  Precision: {rf['precision']:.4f}")
    print(f"  Recall:    {rf['recall']:.4f}")
    print(f"  F1-Score:  {rf['f1_score']:.4f}")
    print(f"  Time:      {rf['training_time']:.2f}s")
    print("--------------------------------------------------")
    print("ADABOOST (BOOSTING):")
    print(f"  Accuracy:  {ada['accuracy']:.4f}")
    print(f"  Precision: {ada['precision']:.4f}")
    print(f"  Recall:    {ada['recall']:.4f}")
    print(f"  F1-Score:  {ada['f1_score']:.4f}")
    print(f"  Time:      {ada['training_time']:.2f}s")
    print("--------------------------------------------------")
    print(f"SELECTED BEST MODEL:    {metadata['selected_model']} (Based on F1-Score)")
    print("==================================================\n")

if __name__ == "__main__":
    train_cli()
