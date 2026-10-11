import time
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model_performance(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Evaluate model performance on test set.
    Returns calculated metrics (accuracy, precision, recall, f1_score, confusion_matrix, timing).
    """
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = round(time.time() - start_pred, 4)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    report = classification_report(y_test, y_pred, output_dict=True)

    training_time = getattr(model, "training_time", 0.0)

    results = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "training_time": training_time,
        "prediction_time": pred_time,
        "classification_report": report
    }
    return results
