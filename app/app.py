"""
app/app.py  — HepatoAI Flask application
Changes from original:
  • Added SHAP TreeExplainer (initialised once at startup, not per-request)
  • /predict route now generates shap_values and passes shap_json to template
  • All original routes and logic preserved unchanged
"""

import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── make sure src/ is importable ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# ── optional: import your config if you have one ────────────────────────────
try:
    from config import MODEL_SAVE_PATH, SCALER_SAVE_PATH
except ImportError:
    MODEL_SAVE_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.pkl")
    SCALER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

# ── NEW: import shap ─────────────────────────────────────────────────────────
import shap

app = Flask(__name__)

# ── Load model & scaler once at startup ─────────────────────────────────────
model  = joblib.load(MODEL_SAVE_PATH)
scaler = joblib.load(SCALER_SAVE_PATH)

# ── NEW: Build SHAP explainer once at startup (slow to build, fast to use) ──
# TreeExplainer is the right choice for Random Forest — exact, not approximate
explainer = shap.TreeExplainer(model)

# ── Feature metadata ─────────────────────────────────────────────────────────
# These are the exact feature names your model was trained on, in order.
# If your order is different, adjust this list to match your training code.
FEATURE_NAMES = [
    "N_Days", "Age", "Sex", "Ascites", "Hepatomegaly",
    "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]

# Human-readable labels shown in the SHAP chart (units help doctors read it)
FEATURE_LABELS = {
    "N_Days":       "Days since diagnosis",
    "Age":          "Age (days)",
    "Sex":          "Sex (0=M, 1=F)",
    "Ascites":      "Ascites (0/1)",
    "Hepatomegaly": "Hepatomegaly (0/1)",
    "Spiders":      "Spider angiomas (0/1)",
    "Edema":        "Edema (0=N, 1=S, 2=Y)",
    "Bilirubin":    "Bilirubin (mg/dL)",
    "Cholesterol":  "Cholesterol (mg/dL)",
    "Albumin":      "Albumin (g/dL)",
    "Copper":       "Copper (µg/day)",
    "Alk_Phos":     "Alkaline Phosphatase (U/L)",
    "SGOT":         "SGOT (U/mL)",
    "Tryglicerides":"Triglycerides (mg/dL)",
    "Platelets":    "Platelets (ml/1000)",
    "Prothrombin":  "Prothrombin time (s)",
}

# Stage labels
STAGE_LABELS = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4"}
STAGE_DESCRIPTIONS = {
    1: "Compensated cirrhosis — liver is scarred but still functions adequately.",
    2: "Moderate scarring with some decline in liver function.",
    3: "Advanced cirrhosis — significant loss of function, complications likely.",
    4: "Decompensated cirrhosis — severe liver failure, urgent care required.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── 1. Parse input values from the form ─────────────────────────────
        raw_values = []
        for feat in FEATURE_NAMES:
            raw_values.append(float(request.form.get(feat, 0)))

        input_array = np.array([raw_values])

        # ── 2. Scale the input (same scaler used during training) ────────────
        input_scaled = scaler.transform(input_array)

        # ── 3. Original prediction ───────────────────────────────────────────
        prediction      = int(model.predict(input_scaled)[0])
        probabilities   = model.predict_proba(input_scaled)[0].tolist()
        confidence      = round(max(probabilities) * 100, 1)

        # ── 4. NEW: SHAP explanation ─────────────────────────────────────────
        shap_values = explainer.shap_values(input_scaled)

        # For multi-class RF, shap_values is a list of arrays (one per class).
        # We show the SHAP values for the predicted class.
        predicted_class_idx = prediction - 1  # stages are 1-indexed, arrays are 0-indexed

        if isinstance(shap_values, list):
            sv_for_class = shap_values[predicted_class_idx][0].tolist()
            base_value   = float(
                explainer.expected_value[predicted_class_idx]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value
            )
        else:
            # Binary fallback
            sv_for_class = shap_values[0].tolist()
            base_value   = float(explainer.expected_value)

        # Build the SHAP payload for the template
        shap_payload = {
            "feature_names":  [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
            "feature_keys":   FEATURE_NAMES,
            "feature_values": [round(v, 3) for v in raw_values],
            "shap_values":    [round(v, 4) for v in sv_for_class],
            "base_value":     round(base_value, 4),
        }

        # ── 5. Render result page ────────────────────────────────────────────
        return render_template(
            "result.html",
            prediction          = prediction,
            stage_label         = STAGE_LABELS.get(prediction, f"Stage {prediction}"),
            stage_description   = STAGE_DESCRIPTIONS.get(prediction, ""),
            confidence          = confidence,
            probabilities       = {STAGE_LABELS[i+1]: round(p*100, 1) for i, p in enumerate(probabilities)},
            shap_json           = json.dumps(shap_payload),   # ← NEW
        )

    except Exception as e:
        return render_template("error.html", error=str(e)), 500


# ── API endpoint (optional — useful if you later add a mobile app) ───────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint — same logic as /predict but returns JSON."""
    try:
        data = request.get_json(force=True)
        raw_values = [float(data.get(f, 0)) for f in FEATURE_NAMES]
        input_array  = np.array([raw_values])
        input_scaled = scaler.transform(input_array)

        prediction    = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0].tolist()
        shap_values   = explainer.shap_values(input_scaled)

        idx = prediction - 1
        sv  = (shap_values[idx][0] if isinstance(shap_values, list) else shap_values[0]).tolist()

        return jsonify({
            "prediction":    prediction,
            "stage_label":   STAGE_LABELS.get(prediction),
            "confidence":    round(max(probabilities) * 100, 1),
            "probabilities": probabilities,
            "shap_values":   {FEATURE_NAMES[i]: round(v, 4) for i, v in enumerate(sv)},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
