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
            customer_id,
            order_id,
            product_id,
            quantity,
            total_amount,
            order_date
        FROM orders
        """

        orders = pd.read_sql_query(query, conn)

    finally:

        conn.close()


    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if orders.empty:

        print("No orders found in the database.")

        return


    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    orders = orders.dropna(
        subset=["customer_id", "order_date"]
    )


    # --------------------------------------------------------
    # SORT ORDERS
    # --------------------------------------------------------

    orders = orders.sort_values(
        ["customer_id", "order_date"]
    )


    # --------------------------------------------------------
    # REFERENCE DATE
    # --------------------------------------------------------

    reference_date = orders["order_date"].max()


    # --------------------------------------------------------
    # CUSTOMER-LEVEL FEATURES
    # --------------------------------------------------------

    features = (
        orders
        .groupby("customer_id")
        .agg(
            recency_days=(
                "order_date",
                lambda x: (
                    reference_date - x.max()
                ).days
            ),

            frequency=(
                "order_id",
                "nunique"
            ),

            total_items=(
                "quantity",
                "sum"
            ),

            total_revenue=(
                "total_amount",
                "sum"
            ),

            unique_products=(
                "product_id",
                "nunique"
            ),

            first_purchase=(
                "order_date",
                "min"
            ),

            last_purchase=(
                "order_date",
                "max"
            )
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # AVERAGE TRANSACTION VALUE
    # --------------------------------------------------------

    features["average_transaction_value"] = (
        features["total_revenue"]
        / features["frequency"]
    )


    # --------------------------------------------------------
    # LIFETIME DAYS
    # --------------------------------------------------------

    features["lifetime_days"] = (
        features["last_purchase"]
        - features["first_purchase"]
    ).dt.days


    # --------------------------------------------------------
    # RETURN COUNT
    # --------------------------------------------------------
    #
    # If the orders table does not contain a return indicator,
    # we use 0 as the default value.
    #

    features["return_count"] = 0


    # --------------------------------------------------------
    # HANDLE POSSIBLE DIVISION ISSUES
    # --------------------------------------------------------

    features["average_transaction_value"] = (
        features["average_transaction_value"]
        .fillna(0)
    )


    # --------------------------------------------------------
    # SELECT MODEL FEATURES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("=" * 60)

    print("Customer feature generation completed.")

    print("=" * 60)

    print(
        f"Customers processed: {len(features)}"
    )

    print(
        f"Features generated: {len(model_features) - 1}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    print("\nGenerated columns:")

    for column in features.columns:

        print(f"- {column}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_customer_features()
