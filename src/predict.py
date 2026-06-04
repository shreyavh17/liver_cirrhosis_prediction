"""
predict.py
----------
Step 6 — Inference / Prediction on new patient data.

Loads the saved Random Forest model and makes predictions on:
  - A single patient (dict input)
  - A CSV file of new patients

Usage:
    python src/predict.py                         ← runs demo example
    python src/predict.py --file new_patients.csv ← batch prediction
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_SAVE_PATH, DATASET_CLEAN_PATH, TARGET_COLUMN
from data_preprocessing import (
    engineer_age_feature, impute_missing_values, encode_categoricals
)

STAGE_LABELS = {
    1: "Stage 1 — Early fibrosis. Minimal symptoms. Good prognosis with treatment.",
    2: "Stage 2 — Moderate fibrosis. Lifestyle changes + medication recommended.",
    3: "Stage 3 — Advanced fibrosis. Regular monitoring required.",
    4: "Stage 4 — Severe cirrhosis (decompensated). Urgent specialist care needed.",
}


def load_model(path: str = MODEL_SAVE_PATH):
    """Load the trained Random Forest model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run main.py first to train and save the model."
        )
    model = joblib.load(path)
    print(f"[MODEL] Loaded Random Forest from → {path}")
    return model


def get_expected_columns() -> list:
    """Read the cleaned dataset to get the exact column order the model expects."""
    df = pd.read_csv(DATASET_CLEAN_PATH)
    return [c for c in df.columns if c != TARGET_COLUMN]


def prepare_patient_input(patient: dict) -> pd.DataFrame:
    """
    Convert a raw patient dict into a model-ready DataFrame
    by aligning columns to what the model was trained on.
    Missing columns → filled with 0.
    """
    expected_cols = get_expected_columns()
    df_patient    = pd.DataFrame([patient])
    df_patient    = df_patient.reindex(columns=expected_cols, fill_value=0)
    return df_patient


def predict_single(patient: dict, model=None) -> dict:
    """
    Predict cirrhosis stage for one patient.

    Parameters
    ----------
    patient : dict  — raw clinical values (e.g. {"Bilirubin": 2.1, "Albumin": 3.5, ...})
    model   : fitted sklearn model or None (loads from disk)

    Returns
    -------
    dict with predicted stage and probability scores
    """
    if model is None:
        model = load_model()

    X = prepare_patient_input(patient)
    stage       = model.predict(X)[0]
    proba       = model.predict_proba(X)[0]
    classes     = model.classes_

    print("\n" + "="*55)
    print("  PREDICTION RESULT")
    print("="*55)
    print(f"  Predicted Stage : {stage}")
    print(f"  Interpretation  : {STAGE_LABELS.get(stage, 'Unknown stage')}")
    print("\n  Probability breakdown:")
    for cls, prob in zip(classes, proba):
        bar = "█" * int(prob * 30)
        print(f"    Stage {cls}  [{bar:<30}]  {prob*100:.1f}%")
    print("="*55 + "\n")

    return {
        "predicted_stage": int(stage),
        "interpretation":  STAGE_LABELS.get(stage, ""),
        "probabilities":   {int(c): round(float(p), 4) for c, p in zip(classes, proba)}
    }


def predict_batch(csv_path: str, model=None) -> pd.DataFrame:
    """
    Predict stages for all patients in a CSV file.
    Saves results to the same directory as the input file.
    """
    if model is None:
        model = load_model()

    df      = pd.read_csv(csv_path)
    X       = df.reindex(columns=get_expected_columns(), fill_value=0)
    stages  = model.predict(X)
    probas  = model.predict_proba(X)

    df["Predicted_Stage"] = stages
    for i, cls in enumerate(model.classes_):
        df[f"Prob_Stage_{cls}"] = probas[:, i].round(4)

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[BATCH] Predictions saved → {out_path}")
    return df


# ── Demo example ──────────────────────────────────────────────────────────────
DEMO_PATIENT = {
    # A sample patient — fill in real values from a blood test report
    "Bilirubin"    : 2.1,
    "Cholesterol"  : 290,
    "Albumin"      : 3.2,
    "Copper"       : 80,
    "Alk_Phos"     : 900,
    "SGOT"         : 100,
    "Tryglicerides": 110,
    "Platelets"    : 200,
    "Prothrombin"  : 11.5,
    "Age_Years"    : 52.0,
    # One-hot encoded fields (set 1 for whichever applies)
    "Status_C"     : 1,   # Censored
    "Status_CL"    : 0,
    "Status_D"     : 0,   # Deceased
    "Drug_D-penicillamine" : 1,
    "Drug_Placebo" : 0,
    "Sex_F"        : 1,
    "Sex_M"        : 0,
    "Ascites_N"    : 1,
    "Ascites_Y"    : 0,
    "Hepatomegaly_N": 1,
    "Hepatomegaly_Y": 0,
    "Spiders_N"    : 1,
    "Spiders_Y"    : 0,
    "Edema_N"      : 1,
    "Edema_S"      : 0,
    "Edema_Y"      : 0,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liver Cirrhosis Stage Predictor")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to CSV file for batch prediction")
    args = parser.parse_args()

    if args.file:
        predict_batch(args.file)
    else:
        print("\n[INFO]  Running demo prediction with sample patient data...")
        predict_single(DEMO_PATIENT)
