PRAGMA foreign_keys = ON;

-- =========================================================
-- CUSTOMERS
-- =========================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    country TEXT,
    first_purchase TEXT,
    last_purchase TEXT,
    total_orders INTEGER,
    total_items INTEGER,
    total_revenue REAL,
    average_transaction_value REAL,
    unique_products INTEGER,
    unique_countries INTEGER,
    return_count INTEGER,
    lifetime_days INTEGER
);


-- =========================================================
-- PRODUCTS
-- =========================================================

CREATE TABLE IF NOT EXISTS products (
    stock_code TEXT PRIMARY KEY,
    description TEXT
);


-- =========================================================
-- ORDERS
-- Transaction-line level data from Online Retail II
-- =========================================================

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT NOT NULL,
    customer_id INTEGER,
    stock_code TEXT,
    invoice_date TEXT,
    quantity INTEGER,
    unit_price REAL,
    country TEXT,
    line_total REAL,
    is_return INTEGER DEFAULT 0,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    FOREIGN KEY (stock_code)
        REFERENCES products(stock_code)
);


-- =========================================================
-- RECOMMENDATIONS
-- =========================================================

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    stock_code TEXT NOT NULL,
    recommendation_score REAL,
    recommendation_method TEXT,
    recommendation_rank INTEGER,
    model_version TEXT,
    created_at TEXT,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    FOREIGN KEY (stock_code)
        REFERENCES products(stock_code)
);


-- =========================================================
-- PREDICTION LOGS
-- =========================================================

CREATE TABLE IF NOT EXISTS prediction_logs (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,

    customer_id INTEGER,

    recency_days REAL,
    frequency REAL,
    total_items REAL,
    total_revenue REAL,
    average_transaction_value REAL,
    unique_products REAL,
    return_count REAL,
    lifetime_days REAL,

    prediction INTEGER,
    repeat_purchase_probability REAL,

    model_name TEXT,
    model_version TEXT,

    timestamp TEXT,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);


-- =========================================================
-- INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_orders_customer
ON orders(customer_id);

CREATE INDEX IF NOT EXISTS idx_orders_product
ON orders(stock_code);

CREATE INDEX IF NOT EXISTS idx_orders_invoice
ON orders(invoice_no);

CREATE INDEX IF NOT EXISTS idx_recommendations_customer
ON recommendations(customer_id);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_customer
ON prediction_logs(customer_id);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
ON prediction_logs(timestamp);
