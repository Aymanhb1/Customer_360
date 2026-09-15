import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = BASE_DIR / "schema.sql"
DATABASE_FILE = BASE_DIR / "retail_customer360.db"


def create_database():
    connection = sqlite3.connect(DATABASE_FILE)

    with open(SCHEMA_FILE, "r", encoding="utf-8") as file:
        schema = file.read()

    connection.executescript(schema)
    connection.commit()
    connection.close()

    print("Database created successfully.")
    print(f"Location: {DATABASE_FILE}")


if __name__ == "__main__":
    create_database()
