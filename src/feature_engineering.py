"""
feature_engineering.py
-----------------------
Step 3 of the ML pipeline.

Responsibilities:
  - Split features (X) and target (y) from the cleaned dataset
  - Apply StandardScaler to numerical features (required for SVM, LR, KNN)
  - Return train/test splits ready for model training
  - Provide the scaler for later use in prediction

Note: Random Forest does NOT need scaling, but we scale for fair
      comparison with SVM and Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_CLEAN_PATH, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE
)


def load_clean_data(path: str = DATASET_CLEAN_PATH) -> pd.DataFrame:
    """Load the preprocessed/cleaned CSV."""
    df = pd.read_csv(path)
    print(f"[LOAD]  Clean data loaded → {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate X (features) and y (target Stage)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"[SPLIT] Features shape: {X.shape}  |  Target shape: {y.shape}")
    print(f"[SPLIT] Feature columns ({len(X.columns)}): {list(X.columns)}")
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on training data only (no data leakage).
    Transform both train and test sets.
    Returns scaled arrays and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrame so column names are preserved
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    print(f"[SCALE] StandardScaler applied — train fit, test transformed")
    return X_train_scaled, X_test_scaled, scaler


def run_feature_engineering(df: pd.DataFrame = None):
    """
    Full feature engineering pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*55)
    print("  STEP 3 — FEATURE ENGINEERING & SPLITTING")
    print("="*55)

    if df is None:
        df = load_clean_data()

    X, y = split_features_target(df)
    feature_names = list(X.columns)

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )
    print(f"[SPLIT] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"[SPLIT] Train class dist: {dict(y_train.value_counts().sort_index())}")
    print(f"[SPLIT] Test  class dist: {dict(y_test.value_counts().sort_index())}")

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    print(f"\n[DONE]  Feature engineering complete")
    print("="*55 + "\n")

    return X_train_s, X_test_s, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run_feature_engineering()
