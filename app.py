import json
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.utils.validation import check_is_fitted


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer 360",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = Path("repeat_purchase_model_v2.pkl")
METADATA_PATH = Path("model_metadata.json")
SCHEMA_PATH = Path("feature_schema.json")

DATABASE_PATH = Path("retail_customer360.db")
SQL_SCHEMA_PATH = Path("schema.sql")


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

@st.cache_data
def load_feature_schema():

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema file not found: {SCHEMA_PATH}"
        )

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    check_is_fitted(model)

    return model


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    # If database already exists, use it
    if DATABASE_PATH.exists():
        return

    # schema.sql must exist
    if not SQL_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Database schema file not found: {SQL_SCHEMA_PATH}"
        )

    # Create database
    conn = sqlite3.connect(DATABASE_PATH)

    try:

        with open(SQL_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)
        conn.commit()

    finally:

        conn.close()


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_connection():

    initialize_database()

    return sqlite3.connect(DATABASE_PATH)


# ============================================================
# SAVE PREDICTION LOG
# ============================================================

def save_prediction_log(log_data):

    conn = get_database_connection()

    try:

        query = """
        INSERT INTO prediction_logs (
            customer_id,
            recency_days,
            frequency,
            total_items,
            total_revenue,
            average_transaction_value,
            unique_products,
            return_count,
            lifetime_days,
            prediction,
            repeat_purchase_probability,
            model_name,
            model_version,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        conn.execute(
            query,
            (
                None,
                float(log_data["recency_days"]),
                float(log_data["frequency"]),
                float(log_data["total_items"]),
                float(log_data["total_revenue"]),
                float(log_data["average_transaction_value"]),
                float(log_data["unique_products"]),
                float(log_data["return_count"]),
                float(log_data["lifetime_days"]),
                int(log_data["prediction"]),
                float(log_data["repeat_purchase_probability"]),
                log_data["model_name"],
                log_data["model_version"],
                log_data["timestamp"]
            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# LOAD MODEL + CONFIGURATION
# ============================================================

try:

    model = load_model()
    metadata = load_metadata()
    feature_schema = load_feature_schema()

except Exception as e:

    st.error("❌ Failed to load application files.")
    st.exception(e)
    st.stop()


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = feature_schema.get("features", [])

if not EXPECTED_FEATURES:

    st.error(
        "❌ No features were found in feature_schema.json."
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_VERSION = metadata.get(
    "model_version",
    feature_schema.get("model_version", "unknown")
)

MODEL_NAME = metadata.get(
    "model_name",
    "Repeat Purchase Prediction Model"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛍️ Customer 360")

st.subheader("Repeat Purchase Prediction")

st.write(
    """
    Enter customer behavioral features below to predict
    whether the customer is likely to make a repeat purchase.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(f"**Model:** {MODEL_NAME}")

    st.write(f"**Version:** {MODEL_VERSION}")

    st.write(
        f"**Expected Features:** {len(EXPECTED_FEATURES)}"
    )

    with st.expander("View Expected Features"):

        for feature in EXPECTED_FEATURES:
            st.write(f"- `{feature}`")


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)


with col1:

    recency_days = st.number_input(
        "Recency Days",
        min_value=0.0,
        value=30.0,
        step=1.0,
        help="Number of days since the customer's last purchase."
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="Number of purchases made by the customer."
    )

    total_items = st.number_input(
        "Total Items",
        min_value=0.0,
        value=10.0,
        step=1.0,
        help="Total number of items purchased."
    )

    total_revenue = st.number_input(
        "Total Revenue",
        min_value=0.0,
        value=500.0,
        step=10.0,
        help="Total revenue generated by the customer."
    )


with col2:

    average_transaction_value = st.number_input(
        "Average Transaction Value",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Average value of the customer's transactions."
    )

    unique_products = st.number_input(
        "Unique Products",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="Number of different products purchased."
    )

    return_count = st.number_input(
        "Return Count",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Number of returned purchases/items."
    )

    lifetime_days = st.number_input(
        "Lifetime Days",
        min_value=0.0,
        value=365.0,
        step=1.0,
        help="Number of days since the customer's first purchase."
    )


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    [
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "total_items": total_items,
            "total_revenue": total_revenue,
            "average_transaction_value": average_transaction_value,
            "unique_products": unique_products,
            "return_count": return_count,
            "lifetime_days": lifetime_days
        }
    ]
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

input_features = set(input_data.columns)

expected_features = set(EXPECTED_FEATURES)

missing_features = expected_features - input_features

unexpected_features = input_features - expected_features


if missing_features:

    st.error(
        "❌ Missing required features: "
        + ", ".join(sorted(missing_features))
    )

    st.stop()


if unexpected_features:

    st.error(
        "❌ Unexpected features found: "
        + ", ".join(sorted(unexpected_features))
    )

    st.stop()


# Ensure exact feature order
input_data = input_data[EXPECTED_FEATURES]


# ============================================================
# DATA VALIDATION
# ============================================================

if input_data.isnull().any().any():

    st.error(
        "❌ Input contains missing values."
    )

    st.stop()


if not all(
    pd.api.types.is_numeric_dtype(input_data[column])
    for column in input_data.columns
):

    st.error(
        "❌ All model features must be numeric."
    )

    st.stop()


if (input_data < 0).any().any():

    st.error(
        "❌ Feature values cannot be negative."
    )

    st.stop()


# ============================================================
# VIEW INPUT
# ============================================================

with st.expander("🔍 View Model Input"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Repeat Purchase",
    type="primary",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            if hasattr(model, "classes_"):

                classes = list(model.classes_)

                if 1 in classes:

                    repeat_purchase_probability = float(
                        probabilities[classes.index(1)]
                    )

                else:

                    repeat_purchase_probability = float(
                        max(probabilities)
                    )

            else:

                repeat_purchase_probability = float(
                    max(probabilities)
                )

        else:

            repeat_purchase_probability = 0.0


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if int(prediction) == 1:

                st.success(
                    "✅ Customer is predicted to make a repeat purchase."
                )

            else:

                st.warning(
                    "⚠️ Customer is predicted NOT to make a repeat purchase."
                )


        with result_col2:

            st.metric(
                "Repeat Purchase Probability",
                f"{repeat_purchase_probability:.2%}"
            )


        # ----------------------------------------------------
        # BUSINESS RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("💡 Business Recommendation")

        if int(prediction) == 1:

            st.write(
                """
                This customer shows characteristics associated with
                repeat purchasing. Consider retention strategies such
                as personalized offers, product recommendations,
                and loyalty rewards.
                """
            )

        else:

            st.write(
                """
                This customer may require additional engagement.
                Consider targeted promotions, personalized
                recommendations, and reactivation campaigns.
                """
            )


        # ----------------------------------------------------
        # LOG DATA
        # ----------------------------------------------------

        log_data = {

            "recency_days": recency_days,

            "frequency": frequency,

            "total_items": total_items,

            "total_revenue": total_revenue,

            "average_transaction_value":
                average_transaction_value,

            "unique_products": unique_products,

            "return_count": return_count,

            "lifetime_days": lifetime_days,

            "prediction": int(prediction),

            "repeat_purchase_probability":
                repeat_purchase_probability,

            "model_name": MODEL_NAME,

            "model_version": MODEL_VERSION,

            "timestamp": datetime.now().isoformat()
        }


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        save_prediction_log(log_data)

        st.success(
            "✅ Prediction successfully logged to database."
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.subheader("📊 Prediction History")


try:

    conn = get_database_connection()


    logs = pd.read_sql_query(
        """
        SELECT *
        FROM prediction_logs
        ORDER BY prediction_id DESC
        LIMIT 20
        """,
        conn
    )


    total_predictions = pd.read_sql_query(
        """
        SELECT COUNT(*) AS total
        FROM prediction_logs
        """,
        conn
    ).iloc[0]["total"]


    conn.close()


    st.write(
        f"Total predictions logged: "
        f"**{int(total_predictions)}**"
    )


    if not logs.empty:

        st.dataframe(
            logs,
            use_container_width=True
        )

    else:

        st.info(
            "No predictions have been logged yet."
        )


except Exception as e:

    st.error(
        "❌ Failed to load prediction history."
    )

    st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Customer 360 | {MODEL_NAME} | Version {MODEL_VERSION}"
)
