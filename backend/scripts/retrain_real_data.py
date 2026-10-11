"""
Full retraining pipeline using the REAL Sentiment140 dataset.
Trains 3 models: Logistic Regression (Baseline), Random Forest (Bagging), AdaBoost (Boosting).
"""
import sys
import time
import argparse
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from app.config import settings
from app.ml.preprocessing import TextPreprocessor
from app.ml.feature_engineering import TFIDFFeatureExtractor
from app.ml.logistic_regression_model import LogisticRegressionSentimentModel
from app.ml.random_forest_model import RandomForestSentimentModel
from app.ml.adaboost_model import AdaBoostSentimentModel
from app.utils.file_utils import save_json
from app.utils.logger import logger


def parse_args():
    p = argparse.ArgumentParser(description="Retrain on real Sentiment140 data")
    p.add_argument("--sample", type=int, default=100000,
                   help="Number of tweets to sample (0 = full dataset)")
    p.add_argument("--full", action="store_true",
                   help="Train on the complete full Sentiment140 dataset")
    return p.parse_args()


def load_real_sentiment140(filepath, sample_size):
    print("\n" + "="*60)
    print("  LOADING REAL SENTIMENT140 DATASET")
    print("="*60)
    print(f"Source file : {filepath}")
    print(f"File size   : {filepath.stat().st_size / 1_000_000:.1f} MB")

    probe = pd.read_csv(filepath, nrows=3, header=None, encoding="latin1")
    if probe.shape[1] != 6:
        raise ValueError(f"Expected 6-column Sentiment140 CSV but got {probe.shape[1]} columns.")
    first_val = str(probe.iloc[0, 0]).strip()
    if first_val not in {"0", "4", "0.0", "4.0"}:
        raise ValueError(f"First column value is '{first_val}' -- expected 0 or 4 for Sentiment140.")
    print(f"Format check: PASSED  (first label value = '{first_val}')")

    print("Reading CSV ... (this may take 30-60 seconds)")
    t0 = time.time()
    df = pd.read_csv(
        filepath,
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"],
        encoding="latin1",
        on_bad_lines="skip",
    )
    print(f"CSV loaded in {time.time() - t0:.1f}s  -- {len(df):,} raw rows")

    df = df[["target", "text"]].copy()
    records_raw = len(df)

    df["target"] = df["target"].map({0: 0, 4: 1})
    invalid_labels = df["target"].isna().sum()
    if invalid_labels > 0:
        print(f"Dropping {invalid_labels:,} rows with unrecognised labels")
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    df["text"] = df["text"].astype(str).str.strip()
    before_clean = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.dropna(subset=["text"])
    dropped_empty = before_clean - len(df)
    if dropped_empty:
        print(f"Dropped {dropped_empty:,} rows with empty/null text")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"])
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes:
        print(f"Dropped {dropped_dupes:,} duplicate tweets")

    df = df.reset_index(drop=True)

    pos = int((df["target"] == 1).sum())
    neg = int((df["target"] == 0).sum())
    print(f"\n  Dataset: Sentiment140")
    print(f"  Records loaded         : {records_raw:,}")
    print(f"  Records after cleaning : {len(df):,}")
    print(f"  Positive (label=1)     : {pos:,}")
    print(f"  Negative (label=0)     : {neg:,}")

    if sample_size and 0 < sample_size < len(df):
        print(f"\n  Sampling {sample_size:,} tweets (stratified) ...")
        half = sample_size // 2
        pos_df = df[df["target"] == 1].sample(n=min(half, pos), random_state=42)
        neg_df = df[df["target"] == 0].sample(n=min(half, neg), random_state=42)
        df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Sample size: {len(df):,} (Development Sample)")
        used_full = False
    else:
        print(f"  Using FULL dataset ({len(df):,} tweets)")
        used_full = True

    final_pos = int((df["target"] == 1).sum())
    final_neg = int((df["target"] == 0).sum())
    print(f"  Final Positive : {final_pos:,}")
    print(f"  Final Negative : {final_neg:,}")
    print("="*60 + "\n")

    df.attrs["used_full"] = used_full
    df.attrs["records_raw"] = records_raw
    return df


def eval_model(model, name, X_test, y_test):
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = round(time.time() - t0, 4)
    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))
    rec  = float(recall_score(y_test,  y_pred, average="binary", pos_label=1, zero_division=0))
    f1   = float(f1_score(y_test,      y_pred, average="binary", pos_label=1, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    training_time = getattr(model, "training_time", 0.0)
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Train time: {training_time:.2f}s   Pred time: {pred_time:.4f}s")
    print(f"    Confusion Matrix: {cm}")
    return {
        "model_name": name,
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "confusion_matrix": cm,
        "training_time":    training_time,
        "prediction_time":  pred_time,
        "classification_report": report,
    }


def main():
    args = parse_args()
    sample_size = 0 if args.full else args.sample
    dataset_path = settings.DATASET_PATH

    if not dataset_path.exists():
        print(f"ERROR: Real Sentiment140 CSV not found at: {dataset_path}")
        sys.exit(1)

    # Step 1: Load real data
    df = load_real_sentiment140(dataset_path, sample_size)
    total_samples = len(df)
    pos_samples   = int((df["target"] == 1).sum())
    neg_samples   = int((df["target"] == 0).sum())
    used_full     = df.attrs.get("used_full", True)
    records_raw   = df.attrs.get("records_raw", total_samples)

    # Step 2: Text preprocessing
    print("STEP 2 -- Preprocessing tweet text ...")
    t0 = time.time()
    preprocessor = TextPreprocessor()
    df["processed_text"] = preprocessor.transform_series(df["text"])
    before_pp = len(df)
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)
    dropped_pp = before_pp - len(df)
    if dropped_pp:
        print(f"  Dropped {dropped_pp:,} rows that became empty after preprocessing")
    print(f"  Preprocessing done in {time.time() - t0:.1f}s  ({len(df):,} usable records)")
    proc_out = settings.PROCESSED_DATA_DIR / "processed_sentiment140.csv"
    df[["text", "target", "processed_text"]].to_csv(proc_out, index=False)
    print(f"  Processed dataset saved to: {proc_out}")

    # Step 3: Train/test split
    print("\nSTEP 3 -- Train/test split (80/20, stratified, random_state=42) ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["processed_text"],
        df["target"],
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=df["target"],
    )
    print(f"  Train : {len(X_train_text):,}  |  Test : {len(X_test_text):,}")

    # Step 4: TF-IDF -- fit on TRAIN only
    print("\nSTEP 4 -- Fitting TF-IDF vectorizer on training data ONLY ...")
    t0 = time.time()
    vectorizer = TFIDFFeatureExtractor()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)
    vectorizer.save()
    n_features = X_train.shape[1]
    print(f"  TF-IDF fitted in {time.time() - t0:.1f}s  ({n_features:,} features)")

    # Step 5: Train Model 1 -- Logistic Regression (Baseline)
    print("\nSTEP 5 -- Training Logistic Regression (Baseline) ...")
    lr = LogisticRegressionSentimentModel()
    lr.fit(X_train, y_train)
    lr.save()
    print(f"  Logistic Regression trained in {lr.training_time:.2f}s")

    # Step 6: Train Model 2 -- Random Forest (Bagging)
    print("\nSTEP 6 -- Training Random Forest (Bagging) ...")
    rf = RandomForestSentimentModel()
    rf.fit(X_train, y_train)
    rf.save()
    print(f"  Random Forest trained in {rf.training_time:.2f}s")

    # Step 7: Train Model 3 -- AdaBoost (Boosting)
    print("\nSTEP 7 -- Training AdaBoost (Boosting) ...")
    ada = AdaBoostSentimentModel()
    ada.fit(X_train, y_train)
    ada.save()
    print(f"  AdaBoost trained in {ada.training_time:.2f}s")

    # Step 8: Evaluate all 3 models
    print("\nSTEP 8 -- Evaluating all 3 models on held-out test set ...")
    lr_metrics  = eval_model(lr,  "Logistic Regression", X_test, y_test)
    rf_metrics  = eval_model(rf,  "Random Forest", X_test, y_test)
    ada_metrics = eval_model(ada, "AdaBoost",       X_test, y_test)

    # Step 9: Dynamic Model selection based on F1 Score
    candidates = [
        ("Logistic Regression", lr_metrics["f1_score"]),
        ("Random Forest", rf_metrics["f1_score"]),
        ("AdaBoost", ada_metrics["f1_score"])
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_model, best_f1 = candidates[0]
    print(f"\n  --> SELECTED MODEL: {selected_model} (F1 = {best_f1:.4f})")

    # Step 10: Save metadata
    training_mode_str = "Full Dataset" if used_full else f"Development Sample ({total_samples:,} tweets)"
    metadata = {
        "dataset_stats": {
            "dataset_name":      "Sentiment140",
            "dataset_source":    str(dataset_path.name),
            "records_raw":       records_raw,
            "total_samples":     total_samples,
            "positive_samples":  pos_samples,
            "negative_samples":  neg_samples,
            "train_size":        int(len(y_train)),
            "test_size":         int(len(y_test)),
            "is_sampled":        not used_full,
            "sample_size_used":  sample_size if (not used_full and sample_size) else total_samples,
            "training_mode":     training_mode_str
        },
        "parameters": {
            "tfidf_max_features":    vectorizer.max_features,
            "tfidf_ngram_range":     list(vectorizer.ngram_range),
            "tfidf_min_df":          vectorizer.min_df,
            "tfidf_max_df":          vectorizer.max_df,
            "tfidf_features_fitted": n_features,
            "test_size":             settings.TEST_SIZE,
            "random_state":          settings.RANDOM_STATE,
            "logistic_max_iter":     lr.max_iter,
            "rf_n_estimators":       rf.n_estimators,
            "rf_max_depth":          rf.max_depth,
            "adaboost_n_estimators": ada.n_estimators,
            "adaboost_learning_rate":ada.learning_rate,
        },
        "models": ["Logistic Regression", "Random Forest", "AdaBoost"],
        "logistic_regression": lr_metrics,
        "random_forest":      rf_metrics,
        "adaboost":           ada_metrics,
        "selected_model":     selected_model,
        "primary_metric":     "f1_score",
    }
    meta_path = settings.MODEL_DIR / "model_metadata.json"
    save_json(metadata, meta_path)
    print(f"\n  Metadata saved to: {meta_path}")

    # Step 11: Dataset report
    print("\n" + "="*60)
    print("  DATASET REPORT & MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"  Dataset            : Sentiment140")
    print(f"  Records loaded     : {records_raw:,}")
    print(f"  Records after clean: {total_samples:,}")
    print(f"  Positive           : {pos_samples:,}")
    print(f"  Negative           : {neg_samples:,}")
    print(f"  Training           : {len(y_train):,}")
    print(f"  Testing            : {len(y_test):,}")
    print(f"  TF-IDF features    : {n_features:,}")
    print(f"  Mode               : {training_mode_str}")
    print("--------------------------------------------------")
    print(f"  Logistic Regression : Acc={lr_metrics['accuracy']:.4f} | F1={lr_metrics['f1_score']:.4f} | TrainTime={lr_metrics['training_time']:.2f}s")
    print(f"  Random Forest       : Acc={rf_metrics['accuracy']:.4f} | F1={rf_metrics['f1_score']:.4f} | TrainTime={rf_metrics['training_time']:.2f}s")
    print(f"  AdaBoost            : Acc={ada_metrics['accuracy']:.4f} | F1={ada_metrics['f1_score']:.4f} | TrainTime={ada_metrics['training_time']:.2f}s")
    print("="*60)

    # Step 12: Manual prediction verification
    print("\n" + "="*60)
    print("  MANUAL PREDICTION VERIFICATION (3-MODEL COMPARISON)")
    print("="*60)

    test_sentences = [
        "I absolutely loved this movie",
        "This movie was amazing",
        "This movie was terrible",
        "I hated this movie",
        "The movie was bad",
        "The movie was good",
    ]

    from app.ml.prediction import PredictionEngine
    engine = PredictionEngine()
    if not engine.load_artifacts():
        print("ERROR: Could not load freshly saved artifacts.")
        sys.exit(1)

    for sent in test_sentences:
        cmp_res = engine.predict_compare(sent)
        print(f"\n  Text : '{cmp_res['text']}'")
        print(f"    - Logistic Reg : {cmp_res['logistic_regression']['sentiment']} ({cmp_res['logistic_regression']['confidence']*100:.1f}%)")
        print(f"    - Random Forest: {cmp_res['random_forest']['sentiment']} ({cmp_res['random_forest']['confidence']*100:.1f}%)")
        print(f"    - AdaBoost     : {cmp_res['adaboost']['sentiment']} ({cmp_res['adaboost']['confidence']*100:.1f}%)")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE -- All 3 models saved.")
    print(f"  Selected Best Model: {selected_model}  (F1 = {best_f1:.4f})")
    print("="*60 + "\n")

    return metadata


if __name__ == "__main__":
    main()
