"""
model_training.py
-----------------
Step 4 of the ML pipeline.

Responsibilities:
  - Dynamically instantiate all 6 classifiers from config.py
  - Train each model on the training set
  - Record training time per model
  - Save the best model (Random Forest) to models/random_forest.pkl
  - Return a dict of trained models
"""

import importlib
import time
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_CONFIG, MODEL_SAVE_PATH


def build_model(model_name: str):
    """
    Dynamically instantiate a sklearn model from MODELS_CONFIG.
    This avoids hardcoding every import.
    """
    cfg    = MODELS_CONFIG[model_name]
    module = importlib.import_module(cfg["module"])
    cls    = getattr(module, cfg["class"])
    model  = cls(**cfg["params"])
    return model


def train_all_models(X_train, y_train) -> dict:
    """
    Train every model defined in MODELS_CONFIG.
    Returns a dict: { model_name: trained_model }
    """
    print("\n" + "="*55)
    print("  STEP 4 — MODEL TRAINING")
    print("="*55)

    trained_models = {}
    timing         = {}

    for name in MODELS_CONFIG:
        print(f"\n  Training: {name} ...")
        model   = build_model(name)
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t_start
        trained_models[name] = model
        timing[name]         = elapsed
        print(f"  ✓  {name} trained in {elapsed:.3f}s")

    print("\n" + "─"*55)
    print("  Training Summary:")
    for name, t in timing.items():
        print(f"    {name:<25} → {t:.3f}s")

    print("="*55 + "\n")
    return trained_models


def save_best_model(trained_models: dict, model_name: str = "Random Forest") -> None:
    """Save the specified model to disk using joblib."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model = trained_models[model_name]
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[SAVE]  '{model_name}' saved → {MODEL_SAVE_PATH}")


def run_model_training(X_train, y_train) -> dict:
    """Full training pipeline. Returns trained_models dict."""
    trained_models = train_all_models(X_train, y_train)
    save_best_model(trained_models)
    return trained_models
