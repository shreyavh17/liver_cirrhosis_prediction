"""
main.py
-------
Master entry point for the Liver Disease Prediction project.

Runs the full ML pipeline in sequence:
  1. Data Preprocessing
  2. Exploratory Data Analysis
  3. Feature Engineering & Train/Test Split
  4. Model Training
  5. Model Evaluation
  6. Demo Prediction

Usage:
    python main.py              ← full pipeline
    python main.py --skip-eda   ← skip EDA (faster re-runs)
    python main.py --eval-only  ← skip training, load saved model & evaluate
"""

import argparse
import joblib
import os
import sys

# ── make sure src/ is on the path ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_preprocessing  import run_preprocessing
from eda                 import run_eda
from feature_engineering import run_feature_engineering
from model_training      import run_model_training
from model_evaluation    import run_evaluation
from predict             import predict_single, load_model, DEMO_PATIENT
from config              import MODEL_SAVE_PATH


def banner():
    print("\n" + "█"*55)
    print("█  LIVER CIRRHOSIS STAGE PREDICTION SYSTEM")
    print("█  Machine Learning Pipeline — Random Forest")
    print("█"*55 + "\n")


def main(skip_eda: bool = False, eval_only: bool = False):
    banner()

    # ── STEP 1: Preprocessing ─────────────────────────────────────────────
    clean_df = run_preprocessing()

    # ── STEP 2: EDA ───────────────────────────────────────────────────────
    if not skip_eda:
        run_eda(clean_df)
    else:
        print("[SKIP]  EDA skipped (--skip-eda flag)\n")

    # ── STEP 3: Feature Engineering ───────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler = \
        run_feature_engineering(clean_df)

    # Save scaler for Flask app
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[SAVE] scaler saved -> models/scaler.pkl")

    # ── STEP 4: Model Training ────────────────────────────────────────────
    if eval_only and os.path.exists(MODEL_SAVE_PATH):
        print("[SKIP]  Training skipped — loading saved model\n")
        rf_model = load_model()
        # Rebuild a minimal dict for evaluation
        trained_models = {"Random Forest": rf_model}
    else:
        trained_models = run_model_training(X_train, y_train)

    # ── STEP 5: Evaluation ────────────────────────────────────────────────
    results_df = run_evaluation(trained_models, X_test, y_test, feature_names)

    # ── STEP 6: Demo Prediction ───────────────────────────────────────────
    print("\n" + "─"*55)
    print("  DEMO — Predicting stage for a sample patient")
    print("─"*55)
    predict_single(DEMO_PATIENT, model=trained_models["Random Forest"])

    # ── Final summary ─────────────────────────────────────────────────────
    best      = results_df.iloc[0]
    print("─"*55)
    print("  PIPELINE COMPLETE ✓")
    print(f"  Best model : {best['Model']}")
    print(f"  Accuracy   : {best['Accuracy']:.2f}%")
    print(f"  Model saved: {MODEL_SAVE_PATH}")
    print(f"  Plots saved: outputs/plots/")
    print(f"  Results CSV: outputs/reports/model_results.csv")
    print("─"*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liver Cirrhosis Prediction Pipeline")
    parser.add_argument("--skip-eda",  action="store_true",
                        help="Skip EDA (faster on re-runs)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and use saved model")
    args = parser.parse_args()

    main(skip_eda=args.skip_eda, eval_only=args.eval_only)
