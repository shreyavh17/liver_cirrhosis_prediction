"""
config.py
---------
Central configuration file for the Liver Disease Prediction project.
All paths, constants, and hyperparameters are defined here.
Change values here only — all other files import from this file.
"""

import os

# ─────────────────────────────────────────────
#  BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_RAW_PATH     = os.path.join(BASE_DIR, "datasets", "raw",       "cirrhosis.csv")
DATASET_CLEAN_PATH   = os.path.join(BASE_DIR, "datasets", "processed", "cirrhosis_cleaned.csv")

MODEL_SAVE_PATH      = os.path.join(BASE_DIR, "models",  "random_forest.pkl")
RESULTS_SAVE_PATH    = os.path.join(BASE_DIR, "outputs", "reports", "model_results.csv")

PLOTS_DIR            = os.path.join(BASE_DIR, "outputs", "plots")

# ─────────────────────────────────────────────
#  DATA SETTINGS
# ─────────────────────────────────────────────
TARGET_COLUMN    = "Stage"
DROP_COLUMNS     = ["ID", "N_Days"]          # columns to drop before modelling
CATEGORICAL_COLS = ["Status", "Drug", "Sex",
                    "Ascites", "Hepatomegaly",
                    "Spiders", "Edema"]

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─────────────────────────────────────────────
#  RANDOM FOREST HYPERPARAMETERS
# ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators" : 200,
    "max_depth"    : None,
    "min_samples_split" : 2,
    "min_samples_leaf"  : 1,
    "random_state"      : RANDOM_STATE,
    "n_jobs"            : -1,
}

# ─────────────────────────────────────────────
#  ALL MODELS TO COMPARE
# ─────────────────────────────────────────────
# (model name → sklearn class + params)
MODELS_CONFIG = {
    "Random Forest": {
        "class"  : "RandomForestClassifier",
        "module" : "sklearn.ensemble",
        "params" : RF_PARAMS,
    },
    "Logis"
    "tic Regression": {
        "class"  : "LogisticRegression",
        "module" : "sklearn.linear_model",
        "params" : {"max_iter": 1000, "random_state": RANDOM_STATE},
    },
    "SVM": {
        "class"  : "SVC",
        "module" : "sklearn.svm",
        "params" : {"random_state": RANDOM_STATE, "probability": True},
    },
    "Gradient Boosting": {
        "class"  : "GradientBoostingClassifier",
        "module" : "sklearn.ensemble",
        "params" : {"n_estimators": 200, "random_state": RANDOM_STATE},
    },
    "KNN": {
        "class"  : "KNeighborsClassifier",
        "module" : "sklearn.neighbors",
        "params" : {"n_neighbors": 5},
    },
    "Decision Tree": {
        "class"  : "DecisionTreeClassifier",
        "module" : "sklearn.tree",
        "params" : {"random_state": RANDOM_STATE},
    },
}
