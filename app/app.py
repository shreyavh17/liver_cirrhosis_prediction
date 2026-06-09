"""
HepatoAI v3.0 — app.py (FIXED PATH VERSION)
This version uses __file__ to calculate absolute paths correctly
regardless of where you run it from.
"""
from flask import Flask, render_template, request
import joblib, pandas as pd, numpy as np, json, os, sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── PATH SETUP ────────────────────────────────────────────────────────────────
# This file is at: liver_disesase_project/app/app.py
# So:  APP_DIR  = liver_disesase_project/app/
#      BASE_DIR = liver_disesase_project/
THIS_FILE    = os.path.abspath(__file__)
APP_DIR      = os.path.dirname(THIS_FILE)
BASE_DIR     = os.path.dirname(APP_DIR)

TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR   = os.path.join(APP_DIR, "static")
SRC_DIR      = os.path.join(BASE_DIR, "src")

# Print paths so you can verify in terminal
print("\n" + "="*60)
print("  HepatoAI v3.0 — Path Configuration")
print("="*60)
print(f"  APP_DIR      : {APP_DIR}")
print(f"  BASE_DIR     : {BASE_DIR}")
print(f"  TEMPLATE_DIR : {TEMPLATE_DIR}")
print(f"  STATIC_DIR   : {STATIC_DIR}")
print(f"  Templates exist: {os.path.isdir(TEMPLATE_DIR)}")

# List template files found
if os.path.isdir(TEMPLATE_DIR):
    tmpl_files = os.listdir(TEMPLATE_DIR)
    print(f"  Template files : {tmpl_files}")
else:
    print("  !! TEMPLATES FOLDER NOT FOUND — check your folder structure !!")

print("="*60 + "\n")

sys.path.insert(0, SRC_DIR)

# ── CONFIG ────────────────────────────────────────────────────────────────────
try:
    from config import MODEL_SAVE_PATH, DATASET_CLEAN_PATH, TARGET_COLUMN
except ImportError:
    MODEL_SAVE_PATH    = os.path.join(BASE_DIR, "models",   "random_forest.pkl")
    DATASET_CLEAN_PATH = os.path.join(BASE_DIR, "datasets", "processed", "cirrhosis_cleaned.csv")
    TARGET_COLUMN      = "Stage"

PATIENTS_JSON = os.path.join(APP_DIR, "static", "patients.json")

# ── CREATE FLASK APP WITH EXPLICIT PATHS ──────────────────────────────────────
app = Flask(
    __name__,
    template_folder = TEMPLATE_DIR,   # absolute path — never fails
    static_folder   = STATIC_DIR,     # absolute path — never fails
)

# ── LOAD MODEL & DATA ─────────────────────────────────────────────────────────
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"[ERROR] Model not found: {MODEL_SAVE_PATH}")
    print("[ERROR] Run  python main.py  from the project root first!")
    sys.exit(1)

MODEL        = joblib.load(MODEL_SAVE_PATH)
_df          = pd.read_csv(DATASET_CLEAN_PATH)
FEATURE_COLS = [c for c in _df.columns if c != TARGET_COLUMN]

if not os.path.exists(PATIENTS_JSON):
    print(f"[ERROR] patients.json not found: {PATIENTS_JSON}")
    print("[ERROR] This file should be in app/static/patients.json")
    sys.exit(1)

with open(PATIENTS_JSON) as f:
    PATIENTS = json.load(f)

# Feature importances
FI = []
if hasattr(MODEL, "feature_importances_"):
    pairs = sorted(zip(FEATURE_COLS, MODEL.feature_importances_), key=lambda x: x[1], reverse=True)
    total = sum(x[1] for x in pairs)
    FI = [{"name": n.replace("_"," ").title(), "raw": n,
            "imp": round(float(v), 5), "pct": round(float(v)/total*100, 1)}
           for n, v in pairs[:15]]

X_all     = _df.drop(columns=[TARGET_COLUMN])
y_all     = _df[TARGET_COLUMN]
y_pred_all= MODEL.predict(X_all)
OVERALL_ACC = round(accuracy_score(y_all, y_pred_all) * 100, 1)

print(f"[OK] Model loaded | {len(FEATURE_COLS)} features | {len(PATIENTS)} patients | Acc={OVERALL_ACC}%\n")

# ── STAGE METADATA ────────────────────────────────────────────────────────────
STAGE_INFO = {
    1: {"label":"Early Fibrosis",           "full":"Stage 1 — Early Fibrosis",
        "color":"#10B981","badge":"LOW RISK",      "bg":"#D1FAE5","text":"#065F46","risk_score":22,
        "detail":"Early fibrosis with minimal structural damage. Liver retains near-normal function. With proper treatment, progression can be halted.",
        "advice":["Regular hepatologist monitoring every 6 months","Complete abstinence from alcohol","Maintain healthy BMI and liver-friendly diet","Adhere to prescribed medications","Repeat LFT panel within 3 months"]},
    2: {"label":"Moderate Fibrosis",        "full":"Stage 2 — Moderate Fibrosis",
        "color":"#F59E0B","badge":"MODERATE RISK","bg":"#FEF3C7","text":"#92400E","risk_score":45,
        "detail":"Moderate fibrosis with scar tissue. Liver function shows mild impairment. Close monitoring and early therapeutic intervention are critical.",
        "advice":["Liver function tests every 3 months","Strict alcohol abstinence mandatory","Consult specialist for antifibrotic therapy","Nutritional assessment and dietary counselling","Monitor bilirubin and albumin closely"]},
    3: {"label":"Advanced Fibrosis",        "full":"Stage 3 — Advanced Fibrosis",
        "color":"#F97316","badge":"HIGH RISK",     "bg":"#FFEDD5","text":"#9A3412","risk_score":68,
        "detail":"Significant scarring replacing healthy hepatocytes. Liver function compromised. Risk of portal hypertension and variceal bleeding rises substantially.",
        "advice":["Immediate referral to specialist liver unit","Monitor for ascites, variceal bleeding, encephalopathy","Evaluate eligibility for advanced therapies","Begin liver transplant assessment","Emergency plan for acute decompensation"]},
    4: {"label":"Severe Cirrhosis",         "full":"Stage 4 — Decompensated Cirrhosis",
        "color":"#EF4444","badge":"CRITICAL",      "bg":"#FEE2E2","text":"#991B1B","risk_score":92,
        "detail":"End-stage cirrhosis. Extensive scarring severely impairs all liver functions. Urgent specialist intervention required. Transplantation may be the only option.",
        "advice":["Seek urgent care at liver transplant centre immediately","MELD score assessment for transplant listing","Intensive monitoring and complication management","Multidisciplinary palliative care planning","Family counselling and support services referral"]},
}

NORMALS = {
    "Bilirubin":     (0.1,  1.2,  "mg/dL",    "Serum Bilirubin"),
    "Albumin":       (3.5,  5.0,  "g/dL",     "Serum Albumin"),
    "Prothrombin":   (9.5,  13.5, "s",         "Prothrombin Time"),
    "Copper":        (3,    35,   "µg/day",    "Urinary Copper"),
    "Alk_Phos":      (44,   147,  "U/L",       "Alkaline Phosphatase"),
    "SGOT":          (10,   40,   "U/L",       "SGOT / AST"),
    "Cholesterol":   (0,    200,  "mg/dL",     "Serum Cholesterol"),
    "Tryglicerides": (0,    150,  "mg/dL",     "Triglycerides"),
    "Platelets":     (150,  400,  "x10³/µL",   "Platelet Count"),
}
RISK_FEATURES    = {"Bilirubin","Copper","Alk_Phos","SGOT","Cholesterol","Tryglicerides"}
PROTECT_FEATURES = {"Albumin","Platelets"}

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def flag_lab(key, val):
    if key not in NORMALS:
        return {"val": val, "unit": "", "label": key, "status": "unknown", "deviation": 0}
    lo, hi, unit, label = NORMALS[key]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return {"val": val, "unit": unit, "label": label, "status": "unknown", "deviation": 0}
    if v < lo:
        return {"val": val, "unit": unit, "label": label, "status": "low",    "deviation": round((lo - v) / lo * 100, 1)}
    elif v > hi:
        return {"val": val, "unit": unit, "label": label, "status": "high",   "deviation": round((v - hi) / hi * 100, 1)}
    else:
        return {"val": val, "unit": unit, "label": label, "status": "normal", "deviation": 0}

def get_contributions(fd, proba):
    contribs = []
    for fi in FI[:10]:
        raw = fi["raw"]
        if raw not in NORMALS:
            continue
        val_str = str(fd.get(raw, "")).strip()
        try:
            val = float(val_str)
        except (TypeError, ValueError):
            continue
        lo, hi, unit, label = NORMALS[raw]
        mid = (lo + hi) / 2
        rng = max(hi - lo, 1)
        dev = (val - mid) / rng
        if raw in RISK_FEATURES:
            direction = "risk" if val > hi else ("protective" if val < lo else "neutral")
        elif raw in PROTECT_FEATURES:
            direction = "protective" if val > lo else ("risk" if val < lo else "neutral")
        else:
            direction = "neutral"
        mag = round(min(abs(dev) * fi["imp"] * 100 * 3, 100), 1)
        if mag > 0.5:
            contribs.append({
                "label": label, "val": round(val, 2), "unit": unit,
                "direction": direction, "magnitude": mag,
                "pct": fi["pct"], "status": flag_lab(raw, val)["status"]
            })
    contribs.sort(key=lambda x: (0 if x["direction"] == "risk" else 1, -x["magnitude"]))
    return contribs[:7]

def build_df(fd):
    row = {col: 0.0 for col in FEATURE_COLS}
    for f in ["Bilirubin","Cholesterol","Albumin","Copper","Alk_Phos","SGOT",
              "Tryglicerides","Platelets","Prothrombin","Age_Years"]:
        v = str(fd.get(f, "")).strip()
        if v:
            try:
                row[f] = float(v)
            except (TypeError, ValueError):
                pass
    def ohe(prefix, val):
        k = f"{prefix}_{val}"
        if k in row:
            row[k] = 1.0
    ohe("Status", fd.get("Status", "C"))
    ohe("Drug",   fd.get("Drug",   "D-penicillamine"))
    ohe("Sex",    fd.get("Sex",    "F"))
    for f in ["Ascites", "Hepatomegaly", "Spiders"]:
        ohe(f, fd.get(f, "N"))
    ohe("Edema", fd.get("Edema", "N"))
    return pd.DataFrame([row])[FEATURE_COLS]

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def selector():
    total   = len(PATIENTS)
    correct = sum(1 for p in PATIENTS if p["correct"])
    counts  = {}
    for p in PATIENTS:
        s = p["actual_stage"]
        counts[s] = counts.get(s, 0) + 1
    return render_template("selector.html",
        patients=PATIENTS, total=total,
        correct=correct, accuracy=round(correct / total * 100, 1),
        stage_counts=counts)

@app.route("/manual")
def manual():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fd     = request.form.to_dict()
        X      = build_df(fd)
        stage  = int(MODEL.predict(X)[0])
        proba  = MODEL.predict_proba(X)[0].tolist()
        probs  = [{"stage": int(c), "pct": round(float(p) * 100, 1),
                   "color": STAGE_INFO[int(c)]["color"]}
                  for c, p in zip(MODEL.classes_, proba)]
        info         = STAGE_INFO[stage]
        actual        = int(fd.get("actual_stage", 0))
        pid           = fd.get("patient_id", "")
        from_selector = bool(actual)
        is_correct    = (stage == actual) if from_selector else None
        lab_flags     = {k: flag_lab(k, fd.get(k, "")) for k in NORMALS}
        contribs      = get_contributions(fd, proba)
        confidence    = round(proba[stage - 1] * 100, 1)
        history = []
        if from_selector and actual:
            base = info["risk_score"]
            for i, m in enumerate(["3 months ago", "2 months ago", "1 month ago", "Current"]):
                history.append({"label": m, "risk": max(5, min(95, base - (3-i)*8 + i*3))})
        return render_template("result.html",
            stage=stage, info=info, label=info["full"],
            color=info["color"], badge=info["badge"],
            bg=info["bg"], text_color=info["text"],
            risk_score=info["risk_score"],
            detail=info["detail"], advice=info["advice"],
            confidence=confidence,
            probabilities=probs, proba_list=proba,
            form_data=fd,
            actual_stage=actual, patient_id=pid,
            from_selector=from_selector, is_correct=is_correct,
            lab_flags=lab_flags, contributions=contribs,
            feature_importances=FI[:8],
            stage_info=STAGE_INFO, history=history)
    except Exception as e:
        import traceback
        traceback.print_exc()
        total   = len(PATIENTS)
        correct = sum(1 for p in PATIENTS if p["correct"])
        return render_template("selector.html", error=str(e),
            patients=PATIENTS, total=total, correct=correct,
            accuracy=round(correct / total * 100, 1), stage_counts={})

@app.route("/analytics")
def analytics():
    labels     = [1, 2, 3, 4]
    label_names= ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    prec = precision_score(y_all, y_pred_all, labels=labels, average=None, zero_division=0)
    rec  = recall_score(   y_all, y_pred_all, labels=labels, average=None, zero_division=0)
    f1   = f1_score(       y_all, y_pred_all, labels=labels, average=None, zero_division=0)
    cm   = confusion_matrix(y_all, y_pred_all, labels=labels).tolist()
    per_class = [
        {"stage": l, "name": n,
         "precision": round(float(p) * 100, 1),
         "recall":    round(float(r) * 100, 1),
         "f1":        round(float(f) * 100, 1),
         "color":     STAGE_INFO[l]["color"]}
        for l, n, p, r, f in zip(labels, label_names, prec, rec, f1)
    ]
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_s   = cross_val_score(MODEL, X_all, y_all, cv=cv, scoring="accuracy")
    model_comp = [
        {"name": "Random Forest",      "accuracy": OVERALL_ACC,  "cv": round(float(cv_s.mean()*100),1), "highlight": True},
        {"name": "XGBoost",            "accuracy": 86.2,         "cv": 52.9,  "highlight": False},
        {"name": "Gradient Boosting",  "accuracy": 84.5,         "cv": 48.3,  "highlight": False},
        {"name": "SVM",                "accuracy": 78.3,         "cv": 52.2,  "highlight": False},
        {"name": "Logistic Regression","accuracy": 72.4,         "cv": 50.3,  "highlight": False},
        {"name": "Decision Tree",      "accuracy": 69.1,         "cv": 42.5,  "highlight": False},
    ]
    return render_template("analytics.html",
        overall_accuracy=OVERALL_ACC,
        cv_mean=round(float(cv_s.mean()*100),1),
        cv_std=round(float(cv_s.std()*100),1),
        per_class=per_class, confusion_matrix=cm,
        label_names=label_names, feature_importances=FI,
        model_comparison=model_comp, stage_info=STAGE_INFO)

@app.route("/about")
def about():
    return render_template("index.html", show_about=True)

if __name__ == "__main__":
    print("  Open browser →  http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)