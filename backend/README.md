# Backend - Sentiment Analysis Ensemble API

FastAPI backend providing TF-IDF feature extraction, Random Forest (Bagging), AdaBoost (Boosting), model evaluation, metadata persistence, single prediction, and CSV batch prediction.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dataset download/prepare script
python scripts/download_dataset.py

# Train models
python scripts/train_models.py

# Evaluate models
python scripts/evaluate_models.py

# Run unit tests
pytest tests/

# Run FastAPI server
uvicorn app.main:app --reload --port 8000
```
