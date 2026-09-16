import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

DATABASE_PATH = Path("retail_customer360.db")


# ============================================================
# LOAD PREDICTION LOGS
# ============================================================

def load_prediction_logs():

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    conn = sqlite3.connect(DATABASE_PATH)

    try:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM prediction_logs
            ORDER BY timestamp
            """,
            conn
        )

    finally:

        conn.close()

    return df


# ============================================================
# MONITORING SUMMARY
# ============================================================

def generate_monitoring_report():

    df = load_prediction_logs()

    print("=" * 60)
    print("CUSTOMER 360 — MODEL MONITORING")
    print("=" * 60)

    if df.empty:

        print("\nNo predictions have been logged yet.")
        return

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    total_predictions = len(df)

    repeat_predictions = (
        df["prediction"] == 1
    ).sum()

    non_repeat_predictions = (
        df["prediction"] == 0
    ).sum()

    repeat_rate = (
        repeat_predictions / total_predictions
    )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    probability_mean = (
        df["repeat_purchase_probability"].mean()
    )

    probability_min = (
        df["repeat_purchase_probability"].min()
    )

    probability_max = (
        df["repeat_purchase_probability"].max()
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_versions = df["model_version"].unique()

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print(f"\nTotal predictions: {total_predictions}")

    print(
        f"Predicted repeat purchases: "
        f"{repeat_predictions}"
    )

    print(
        f"Predicted non-repeat purchases: "
        f"{non_repeat_predictions}"
    )

    print(
        f"Predicted repeat rate: "
        f"{repeat_rate:.2%}"
    )

    print(
        f"\nAverage repeat probability: "
        f"{probability_mean:.4f}"
    )

    print(
        f"Minimum probability: "
        f"{probability_min:.4f}"
    )

    print(
        f"Maximum probability: "
        f"{probability_max:.4f}"
    )

    print(
        f"\nModel versions observed: "
        f"{list(model_versions)}"
    )

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_monitoring_report()
