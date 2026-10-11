import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
from app.utils.logger import logger

def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved joblib artifact to {path}")

def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model/Artifact file not found at {path}")
    logger.info(f"Loading joblib artifact from {path}")
    return joblib.load(path)

def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON metadata to {path}")

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect text and target columns in a DataFrame.
    """
    text_col = None
    target_col = None

    # Check column names (lowercased)
    lower_cols = [str(c).lower().strip() for c in df.columns]
    
    # Text column heuristics
    text_candidates = ["text", "tweet", "comment", "review", "sentence", "content"]
    for cand in text_candidates:
        if cand in lower_cols:
            text_col = df.columns[lower_cols.index(cand)]
            break

    if not text_col:
        # Fallback to column with longest string values or last string column
        for col in reversed(df.columns):
            if df[col].dtype == "object":
                text_col = col
                break
    if not text_col:
        text_col = df.columns[-1]

    # Target column heuristics
    target_candidates = ["target", "sentiment", "label", "class", "category"]
    for cand in target_candidates:
        if cand in lower_cols:
            target_col = df.columns[lower_cols.index(cand)]
            break

    if not target_col:
        # Fallback to first numeric or binary column other than text_col
        for col in df.columns:
            if col != text_col:
                target_col = col
                break
    if not target_col:
        target_col = df.columns[0]

    logger.info(f"Detected columns - Text: '{text_col}', Target: '{target_col}'")
    return str(text_col), str(target_col)

def load_and_standardize_dataset(filepath: Path, sample_size: int = None) -> pd.DataFrame:
    """
    Load dataset from CSV file, handle Sentiment140 6-column headerless format or standard CSV with headers.
    Map labels to 0 (Negative) and 1 (Positive).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    logger.info(f"Loading raw dataset from {filepath}")
    
    # Try reading first line to determine if it has header or standard Sentiment140 structure
    sample_df = pd.read_csv(filepath, nrows=5, encoding="latin1", header=None, on_bad_lines="skip")
    
    if sample_df.shape[1] == 6 and (str(sample_df.iloc[0, 0]).strip() in ["0", "4", "0.0", "4.0"]):
        # Headerless 6-column Sentiment140 format: target, ids, date, flag, user, text
        logger.info("Detected standard headerless Sentiment140 format (6 columns).")
        try:
            df = pd.read_csv(
                filepath,
                encoding="latin1",
                header=None,
                names=["target", "ids", "date", "flag", "user", "text"],
                on_bad_lines="skip"
            )
        except Exception:
            df = pd.read_csv(
                filepath,
                encoding="latin1",
                header=None,
                names=["target", "ids", "date", "flag", "user", "text"],
                on_bad_lines="skip",
                engine="python"
            )
        text_col, target_col = "text", "target"
    else:
        # Standard CSV with headers
        try:
            df = pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip", engine="python")
        text_col, target_col = detect_columns(df)

    # Standardize column names
    df = df[[text_col, target_col]].rename(columns={text_col: "text", target_col: "target"})

    # Clean missing text or targets
    df = df.dropna(subset=["text", "target"])
    df["text"] = df["text"].astype(str)

    # Standardize target values (0 = Negative, 4 -> 1, 1 = Positive, 'positive'/'negative' strings)
    def normalize_target(val):
        str_val = str(val).strip().lower()
        if str_val in ["4", "1", "positive", "pos", "1.0"]:
            return 1
        elif str_val in ["0", "negative", "neg", "0.0"]:
            return 0
        else:
            try:
                num = float(val)
                return 1 if num > 0 else 0
            except ValueError:
                return 0

    df["target"] = df["target"].apply(normalize_target)
    
    # Drop empty texts after stripping whitespace
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0]

    # Drop duplicate text entries
    df = df.drop_duplicates(subset=["text"])

    if sample_size and sample_size > 0 and sample_size < len(df):
        logger.info(f"Sampling {sample_size} rows from dataset (total available: {len(df)}) with stratification.")
        half = sample_size // 2
        pos_df = df[df["target"] == 1]
        neg_df = df[df["target"] == 0]
        n_pos = min(len(pos_df), half)
        n_neg = min(len(neg_df), sample_size - n_pos)
        sampled_pos = pos_df.sample(n=n_pos, random_state=42)
        sampled_neg = neg_df.sample(n=n_neg, random_state=42)
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded dataset: {len(df)} records. Positive: {(df['target']==1).sum()}, Negative: {(df['target']==0).sum()}")
    return df
