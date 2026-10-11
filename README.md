# Sentiment Analysis using Ensemble Learning

An end-to-end Machine Learning web application that classifies text sentiment (Positive vs. Negative) on the Sentiment140 dataset using **Bagging (Random Forest)** and **Boosting (AdaBoost)** algorithms.

---

## Abstract

Sentiment Analysis is a vital task in Natural Language Processing (NLP) that automatically extracts subjective opinion from text data. Traditional single-model machine learning classifiers often suffer from high variance or high bias when trained on high-dimensional text feature spaces. This project presents a full-stack academic machine learning application comparing two major ensemble learning paradigms: **Random Forest (Bagging)** and **AdaBoost (Boosting)**. Utilizing TF-IDF feature representations extracted from preprocessed Sentiment140 tweets, both classifiers are trained, evaluated, and compared side-by-side across key classification metrics (Accuracy, Precision, Recall, and F1-score).

---

## Problem Statement

Single Machine Learning classifiers (such as standard decision trees or single Naive Bayes models) struggle to maintain high accuracy when dealing with noisy, unstructured, or informal microblogging text (such as tweets). Errors in single classifiers directly impact prediction performance. Furthermore, high-dimensional TF-IDF sparse matrices can lead to overfitting in standalone models.

---

## Objectives

1. Preprocess raw text from the Sentiment140 dataset using NLTK (lowercasing, URL removal, mention cleaning, stopword filtering with negation preservation, and lemmatization).
2. Transform text documents into numerical feature vectors using TF-IDF.
3. Train a **Random Forest Classifier** representing **Bagging (Bootstrap Aggregating)**.
4. Train an **AdaBoost Classifier** representing **Boosting (Sequential Error Reweighting)**.
5. Evaluate both models independently on a stratified 20% test split.
6. Compare performance across Accuracy, Precision, Recall, F1-Score, and execution timing.
7. Automatically select the top model based on **F1-Score**.
8. Provide a full-stack application featuring a FastAPI REST API and React/Vite/TypeScript dashboard for real-time and CSV batch predictions.

---

## Existing System vs. Proposed System

### Existing System
- Relies on single classifiers (e.g. single decision tree).
- Highly sensitive to noisy text variations.
- Lack of interactive model comparison dashboards or batch processing tools.

### Proposed System
- Implements Ensemble Learning methods:
  - **Bagging (Random Forest):** Combines parallel decorrelated decision trees to reduce variance.
  - **Boosting (AdaBoost):** Sequentially reweights misclassified samples to reduce bias.
- Provides real-time inference with probability confidence scores.
- Includes a interactive web dashboard, batch CSV processing, and automated persistence via Joblib.

---

## Methodology

```
                   Sentiment140 Dataset (1.6M Tweets)
                                   ↓
                         Text Data Preprocessing
       (Lowercasing, Mention Removal, Stopwords + Negation, Lemmatization)
                                   ↓
                       TF-IDF Feature Extraction
                     (N-gram Range: 1-2, Max Features)
                                   ↓
                       Stratified Train/Test Split
                            (80% Train, 20% Test)
                                   ↓
                  ┌─────────────────────────────────┐
                  │                                 │
        Random Forest (Bagging)             AdaBoost (Boosting)
                  │                                 │
                  └────────────────┬────────────────┘
                                   ↓
                      Independent Model Evaluation
              (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
                                   ↓
                         Automated Model Selection
                         (Highest F1-Score Selected)
                                   ↓
                         FastAPI & React Dashboard
                       (Single & Batch CSV Predictions)
```

---

## Dataset

- **Name:** Sentiment140 Dataset (Stanford / Kaggle)
- **Size:** 1.6 Million tweets
- **Class Balance:** 800,000 Positive (Label 1), 800,000 Negative (Label 0)
- **Format:** CSV with Target, ID, Date, Flag, User, and Tweet Text columns.

---

## Technologies Used

### Backend & Machine Learning
- **Python 3.11+**
- **FastAPI & Uvicorn**
- **Scikit-Learn**
- **Pandas & NumPy**
- **NLTK (Natural Language Toolkit)**
- **Joblib**
- **Pytest**

### Frontend Dashboard
- **React 18 & Vite**
- **TypeScript**
- **Tailwind CSS v4**
- **Recharts**
- **Lucide React Icons**
- **Axios**

---

## Project Architecture

```
sentiment-analysis-ensemble/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── routes_prediction.py
│   │   │   ├── routes_training.py
│   │   │   ├── routes_dataset.py
│   │   │   └── routes_metrics.py
│   │   ├── ml/
│   │   │   ├── preprocessing.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── random_forest_model.py
│   │   │   ├── adaboost_model.py
│   │   │   ├── ensemble_manager.py
│   │   │   ├── evaluation.py
│   │   │   └── prediction.py
│   │   ├── schemas/
│   │   └── utils/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── random_forest.joblib
│   │   ├── adaboost.joblib
│   │   └── model_metadata.json
│   ├── scripts/
│   │   ├── download_dataset.py
│   │   ├── preprocess_dataset.py
│   │   ├── train_models.py
│   │   └── evaluate_models.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## Quickstart & Installation

### 1. Backend Setup

```bash
# Navigate to workspace root
python -m venv venv
.\venv\Scripts\activate  # Windows (or source venv/bin/activate on Linux/Mac)

# Install backend dependencies
pip install -r backend/requirements.txt

# Download dataset & train ensemble models
python backend/scripts/download_dataset.py
python backend/scripts/train_models.py

# Start FastAPI server
uvicorn backend.app.main:app --reload --port 8000
```

FastAPI Swagger Documentation will be accessible at: `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The React dashboard will run locally at: `http://localhost:5173`

---

## Running with Docker Compose

To launch both the backend API and frontend dashboard using Docker:

```bash
docker compose up --build
```

- **Frontend Dashboard:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`

---

## Model Evaluation Results

Actual metrics obtained from executed evaluation on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Bagging)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.68s** |
| **AdaBoost (Boosting)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.97s** |

**Selected Model:** Random Forest (Selected based on highest F1-score & faster training speed).

---

## API Endpoints Summary

- `GET /`: Health check & API details
- `GET /api/health`: Model loading status
- `GET /api/dataset/stats`: Dataset statistics and sample records
- `GET /api/models`: Available models list
- `GET /api/metrics`: Model evaluation metrics
- `GET /api/model-comparison`: Side-by-side model comparison
- `POST /api/predict`: Single text sentiment prediction
- `POST /api/predict/batch`: Batch CSV sentiment analysis
- `POST /api/train`: Trigger background model re-training
- `GET /api/training/status`: Check live training progress

---

## Future Scope

1. Integration of Deep Learning models (e.g. BERT or RoBERTa) as benchmark baselines.
2. Expanding multi-class sentiment detection (Positive, Negative, Neutral).
3. Streaming Twitter API integration for live sentiment monitoring.

---

## Team Members & Academic Attribution

Developed as an academic Machine Learning & Full-Stack Engineering project.
