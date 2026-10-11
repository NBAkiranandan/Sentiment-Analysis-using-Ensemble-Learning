import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ml.preprocessing import TextPreprocessor

def test_clean_text():
    preprocessor = TextPreprocessor()
    raw = "Check out this AWESOME link https://example.com and follow @user123 #happy!!!"
    cleaned = preprocessor.clean_text(raw)
    
    assert "https" not in cleaned
    assert "user123" not in cleaned
    assert "awesome" in cleaned
    assert "happy" in cleaned

def test_preserves_negation():
    preprocessor = TextPreprocessor()
    raw = "I do not like this terrible service at all."
    processed = preprocessor.tokenize_and_filter(raw)
    
    assert "not" in processed
    assert "like" in processed or "terrible" in processed
