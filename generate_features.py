import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = Path("retail_customer360.db")
OUTPUT_PATH = Path("customer_features.csv")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    return sqlite3.connect(DATABASE_PATH)


# ============================================================
# GENERATE CUSTOMER FEATURES
# ============================================================

def generate_customer_features():

    conn = get_connection()

    try:

        query = """
        SELECT
            order_id,
            customer_id,
            stock_code,
            invoice_date,
            quantity,
            line_total,
            is_return
        FROM orders
        """

        orders = pd.read_sql_query(
            query,
            conn
        )

    finally:

        conn.close()


    # ========================================================
    # CHECK DATA
    # ========================================================

    if orders.empty:

        print("No orders found in the database.")

        return


    # ========================================================
    # CONVERT DATE
    # ========================================================

    orders["invoice_date"] = pd.to_datetime(
        orders["invoice_date"],
        errors="coerce"
    )


    # Remove rows with invalid customer/date
    orders = orders.dropna(
        subset=[
            "customer_id",
            "invoice_date"
        ]
    )


    # ========================================================
    # SORT ORDERS
    # ========================================================

    orders = orders.sort_values(
        [
            "customer_id",
            "invoice_date"
        ]
    )


    # ========================================================
    # REFERENCE DATE
    # ========================================================

    reference_date = orders["invoice_date"].max()


    # ========================================================
    # CUSTOMER-LEVEL FEATURES
    # ========================================================

    features = (
        orders
        .groupby("customer_id")
        .agg(

            # Days since customer's latest purchase
            recency_days=(
                "invoice_date",
                lambda x: (
                    reference_date - x.max()
                ).days
            ),

            # Number of unique invoices
            frequency=(
                "invoice_date",
                "nunique"
            ),

            # Total number of items purchased
            total_items=(
                "quantity",
                "sum"
            ),

            # Total revenue
            total_revenue=(
                "line_total",
                "sum"
            ),

            # Number of different products
            unique_products=(
                "stock_code",
                "nunique"
            ),

            # Number of returned lines
            return_count=(
                "is_return",
                "sum"
            ),

            # First purchase date
            first_purchase=(
                "invoice_date",
                "min"
            ),

            # Last purchase date
            last_purchase=(
                "invoice_date",
                "max"
            )
        )
        .reset_index()
    )


    # ========================================================
    # AVERAGE TRANSACTION VALUE
    # ========================================================

    features["average_transaction_value"] = (
        features["total_revenue"]
        / features["frequency"]
    )


    # ========================================================
    # LIFETIME DAYS
    # ========================================================

    features["lifetime_days"] = (
        features["last_purchase"]
        - features["first_purchase"]
    ).dt.days


    # ========================================================
    # HANDLE MISSING / INVALID VALUES
    # ========================================================

    features["average_transaction_value"] = (
        features["average_transaction_value"]
        .fillna(0)
    )

    features["lifetime_days"] = (
        features["lifetime_days"]
        .fillna(0)
    )

    features["return_count"] = (
        features["return_count"]
        .fillna(0)
    )


    # ========================================================
    # SELECT MODEL FEATURES
    # ========================================================

    model_features = [

        "customer_id",

        "recency_days",

        "frequency",

        "total_items",

        "total_revenue",

        "average_transaction_value",

        "unique_products",

        "return_count",

        "lifetime_days"
    ]


    features = features[model_features]


    # ========================================================
    # SAVE FEATURES
    # ========================================================

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print("=" * 60)

    print(
        "Customer feature generation completed successfully."
    )

    print("=" * 60)

    print(
        f"Customers processed: {len(features)}"
    )

    print(
        f"Model features generated: {len(model_features) - 1}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    print("\nGenerated columns:")

    for column in features.columns:

        print(f"- {column}")

    print("\nFirst 5 customers:")

    print(
        features.head().to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_customer_features()
