import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ml.prediction import prediction_engine

def test_prediction_single_valid():
    prediction_engine.load_artifacts()
    if not prediction_engine.is_loaded:
        pytest.skip("Models not trained yet.")
    res = prediction_engine.predict_single("I absolutely love this amazing product!")
    assert res["sentiment"] in ["Positive", "Negative"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res

def test_prediction_three_models():
    prediction_engine.load_artifacts()
    if not prediction_engine.is_loaded:
        pytest.skip("Models not trained yet.")
    res_lr = prediction_engine.predict_single("Super happy with the service.", model_choice="Logistic Regression")
    res_rf = prediction_engine.predict_single("Super happy with the service.", model_choice="Random Forest")
    res_ada = prediction_engine.predict_single("Super happy with the service.", model_choice="AdaBoost")
    
    assert res_lr["model"] == "Logistic Regression"
    assert res_rf["model"] == "Random Forest"
    assert res_ada["model"] == "AdaBoost"

def test_prediction_compare():
    prediction_engine.load_artifacts()
    if not prediction_engine.is_loaded:
        pytest.skip("Models not trained yet.")
    res = prediction_engine.predict_compare("Great experience overall!")
    assert "logistic_regression" in res
    assert "random_forest" in res
    assert "adaboost" in res
    assert res["logistic_regression"]["sentiment"] in ["Positive", "Negative"]
    assert res["random_forest"]["sentiment"] in ["Positive", "Negative"]
    assert res["adaboost"]["sentiment"] in ["Positive", "Negative"]

def test_empty_text_error():
    prediction_engine.load_artifacts()
    if not prediction_engine.is_loaded:
        pytest.skip("Models not trained yet.")
    with pytest.raises(ValueError):
        prediction_engine.predict_single("   ")
