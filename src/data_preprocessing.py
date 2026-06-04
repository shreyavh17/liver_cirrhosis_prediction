"""
data_preprocessing.py
---------------------
Step 1 of the ML pipeline.

Responsibilities:
  - Load raw CSV
  - Drop rows with missing target (Stage)
  - Drop irrelevant columns (ID, N_Days)
  - Impute missing numerical values with column mean
  - Impute missing categorical values with column mode
  - One-hot encode categorical columns
  - Save cleaned dataset to datasets/processed/
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_RAW_PATH, DATASET_CLEAN_PATH,
    TARGET_COLUMN, DROP_COLUMNS, CATEGORICAL_COLS
)


def load_raw_data(path: str = DATASET_RAW_PATH) -> pd.DataFrame:
    """Load the raw cirrhosis CSV file."""
    df = pd.read_csv(path)
    print(f"[LOAD]  Raw data loaded → {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_invalid_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where Stage is NaN or 'NA' (these 6 rows are
    patients without a recorded stage — not useful for supervised learning).
    """
    before = len(df)
    df = df[df[TARGET_COLUMN].notna()]
    df = df[df[TARGET_COLUMN].astype(str).str.strip() != "NA"]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    after = len(df)
    print(f"[DROP]  Removed {before - after} rows with missing Stage → {after} rows remain")
    return df.reset_index(drop=True)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and N_Days (ID is irrelevant; N_Days is used to derive Age)."""
    existing = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[DROP]  Dropped columns: {existing}")
    return df


def engineer_age_feature(df: pd.DataFrame, n_days_col: str = "N_Days") -> pd.DataFrame:
    """
    Derive Age in years from N_Days before dropping N_Days.
    Called BEFORE drop_unnecessary_columns if N_Days is still present.
    """
    if n_days_col in df.columns:
        df["Age_Years"] = (df[n_days_col] / 365.25).round(1)
        print(f"[FEAT]  'Age_Years' derived from N_Days")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical columns  → mean
      - Categorical columns → mode (most frequent)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        if col == TARGET_COLUMN:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            print(f"[IMPUTE] '{col}': {n_missing} missing → filled with mean ({fill_val:.2f})")

    for col in cat_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            print(f"[IMPUTE] '{col}': {n_missing} missing → filled with mode ('{fill_val}')")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns present in CATEGORICAL_COLS.
    drop_first=False so all categories are visible (better interpretability).
    """
    cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    print(f"[ENCODE] One-hot encoded: {cols_to_encode}")
    print(f"[ENCODE] Dataset shape after encoding → {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, path: str = DATASET_CLEAN_PATH) -> None:
    """Save the cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE]  Cleaned data saved → {path}")


def run_preprocessing() -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline in order.
    Returns the cleaned DataFrame.
    """
    print("\n" + "="*55)
    print("  STEP 1 — DATA PREPROCESSING")
    print("="*55)

    df = load_raw_data()
    df = engineer_age_feature(df)       # derive Age before dropping N_Days
    df = drop_invalid_targets(df)
    df = drop_unnecessary_columns(df)
    df = impute_missing_values(df)
    df = encode_categoricals(df)
    save_processed_data(df)

    print(f"\n[DONE]  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("="*55 + "\n")
    return df


if __name__ == "__main__":
    run_preprocessing()
