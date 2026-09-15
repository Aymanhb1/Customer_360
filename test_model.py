import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.utils.validation import check_is_fitted


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "repeat_purchase_model_v2.pkl"

METADATA_PATH = ROOT_DIR / "model_metadata.json"


# ============================================================
# LOAD METADATA
# ============================================================

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)


EXPECTED_FEATURES = metadata["features"]


# ============================================================
# TEST 1 — MODEL FILE EXISTS
# ============================================================

def test_model_file_exists():

    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}"
    )


# ============================================================
# TEST 2 — MODEL CAN BE LOADED
# ============================================================

def test_model_can_be_loaded():

    model = joblib.load(MODEL_PATH)

    assert model is not None


# ============================================================
# TEST 3 — MODEL IS FITTED
# ============================================================

def test_model_is_fitted():

    model = joblib.load(MODEL_PATH)

    check_is_fitted(model)


# ============================================================
# TEST 4 — FEATURE SCHEMA
# ============================================================

def test_feature_schema():

    expected_features = [
        "recency_days",
        "frequency",
        "total_items",
        "total_revenue",
        "average_transaction_value",
        "unique_products",
        "return_count",
        "lifetime_days"
    ]

    assert EXPECTED_FEATURES == expected_features


# ============================================================
# TEST 5 — MODEL ACCEPTS CORRECT INPUT
# ============================================================

def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    test_data = pd.DataFrame({

        "recency_days": [30.0],

        "frequency": [5.0],

        "total_items": [250.0],

        "total_revenue": [1000.0],

        "average_transaction_value": [200.0],

        "unique_products": [20.0],

        "return_count": [0.0],

        "lifetime_days": [180.0]

    })

    prediction = model.predict(test_data)

    assert len(prediction) == 1


# ============================================================
# TEST 6 — PREDICTION IS VALID
# ============================================================

def test_prediction_value():

    model = joblib.load(MODEL_PATH)

    test_data = pd.DataFrame({

        "recency_days": [30.0],

        "frequency": [5.0],

        "total_items": [250.0],

        "total_revenue": [1000.0],

        "average_transaction_value": [200.0],

        "unique_products": [20.0],

        "return_count": [0.0],

        "lifetime_days": [180.0]

    })

    prediction = model.predict(test_data)[0]

    assert prediction in [0, 1]


# ============================================================
# TEST 7 — PROBABILITY IS VALID
# ============================================================

def test_prediction_probability():

    model = joblib.load(MODEL_PATH)

    test_data = pd.DataFrame({

        "recency_days": [30.0],

        "frequency": [5.0],

        "total_items": [250.0],

        "total_revenue": [1000.0],

        "average_transaction_value": [200.0],

        "unique_products": [20.0],

        "return_count": [0.0],

        "lifetime_days": [180.0]

    })

    probability = model.predict_proba(test_data)[0][1]

    assert 0.0 <= probability <= 1.0