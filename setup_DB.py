"""
Database Setup Script for TPC-H Benchmark

This script creates all TPC-H tables, adds primary/foreign keys,
and imports CSV data into PostgreSQL.

Usage:
    python setup_db.py

Prerequisites:
    - PostgreSQL is installed and running
    - Database 'tpch' exists (createdb tpch)
    - CSV files are in Data/_0.125/ directory
"""

import psycopg2
import os
import sys

# ==================== Configuration ====================
DB_NAME = "tpch"
DB_USER = "postgres"
DB_PASSWORD = "your password" 
DB_HOST = "localhost"
DB_PORT = "5432"

DATA_SCALE = "0.125"
DATA_DIR = f"Data/_{DATA_SCALE}"

# TPC-H tables in order (child tables first for deletion)
TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]

# ==================== SQL Definitions ====================

CREATE_TABLE_SQL = {
    "region": """
        CREATE TABLE region (
            r_regionkey INTEGER NOT NULL,
            r_name CHAR(25) NOT NULL,
            r_comment VARCHAR(152)
        )
    """,

    "nation": """
        CREATE TABLE nation (
            n_nationkey INTEGER NOT NULL,
            n_name CHAR(25) NOT NULL,
            n_regionkey INTEGER NOT NULL,
            n_comment VARCHAR(152)
        )
    """,

    "supplier": """
        CREATE TABLE supplier (
            s_suppkey INTEGER NOT NULL,
            s_name CHAR(25) NOT NULL,
            s_address VARCHAR(40) NOT NULL,
            s_nationkey INTEGER NOT NULL,
            s_phone CHAR(15) NOT NULL,
            s_acctbal DECIMAL(15,2) NOT NULL,
            s_comment VARCHAR(101)
        )
    """,

    "customer": """
        CREATE TABLE customer (
            c_custkey INTEGER NOT NULL,
            c_name VARCHAR(25) NOT NULL,
            c_address VARCHAR(40) NOT NULL,
            c_nationkey INTEGER NOT NULL,
            c_phone CHAR(15) NOT NULL,
            c_acctbal DECIMAL(15,2) NOT NULL,
            c_mktsegment CHAR(10) NOT NULL,
            c_comment VARCHAR(117)
        )
    """,

    "part": """
        CREATE TABLE part (
            p_partkey INTEGER NOT NULL,
            p_name VARCHAR(55) NOT NULL,
            p_mfgr CHAR(25) NOT NULL,
            p_brand CHAR(10) NOT NULL,
            p_type VARCHAR(25) NOT NULL,
            p_size INTEGER NOT NULL,
            p_container CHAR(10) NOT NULL,
            p_retailprice DECIMAL(15,2) NOT NULL,
            p_comment VARCHAR(23)
        )
    """,

    "partsupp": """
        CREATE TABLE partsupp (
            ps_partkey INTEGER NOT NULL,
            ps_suppkey INTEGER NOT NULL,
            ps_availqty INTEGER NOT NULL,
            ps_supplycost DECIMAL(15,2) NOT NULL,
            ps_comment VARCHAR(199)
        )
    """,

    "orders": """
        CREATE TABLE orders (
            o_orderkey INTEGER NOT NULL,
            o_custkey INTEGER NOT NULL,
            o_orderstatus CHAR(1) NOT NULL,
            o_totalprice DECIMAL(15,2) NOT NULL,
            o_orderdate DATE NOT NULL,
            o_orderpriority CHAR(15) NOT NULL,
            o_clerk CHAR(15) NOT NULL,
            o_shippriority INTEGER NOT NULL,
            o_comment VARCHAR(79)
        )
    """,

    "lineitem": """
        CREATE TABLE lineitem (
            l_orderkey INTEGER NOT NULL,
            l_partkey INTEGER NOT NULL,
            l_suppkey INTEGER NOT NULL,
            l_linenumber INTEGER NOT NULL,
            l_quantity DECIMAL(15,2) NOT NULL,
            l_extendedprice DECIMAL(15,2) NOT NULL,
            l_discount DECIMAL(15,2) NOT NULL,
            l_tax DECIMAL(15,2) NOT NULL,
            l_returnflag CHAR(1) NOT NULL,
            l_linestatus CHAR(1) NOT NULL,
            l_shipdate DATE NOT NULL,
            l_commitdate DATE NOT NULL,
            l_receiptdate DATE NOT NULL,
            l_shipinstruct CHAR(25) NOT NULL,
            l_shipmode CHAR(10) NOT NULL,
            l_comment VARCHAR(44)
        )
    """
}

ADD_PRIMARY_KEYS = [
    "ALTER TABLE region ADD PRIMARY KEY (r_regionkey)",
    "ALTER TABLE nation ADD PRIMARY KEY (n_nationkey)",
    "ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey)",
    "ALTER TABLE customer ADD PRIMARY KEY (c_custkey)",
    "ALTER TABLE part ADD PRIMARY KEY (p_partkey)",
    "ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey)",
    "ALTER TABLE orders ADD PRIMARY KEY (o_orderkey)",
    "ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber)"
]

ADD_FOREIGN_KEYS = [
    "ALTER TABLE nation ADD FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey)",
    "ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey)",
    "ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey)",
    "ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey)",
    "ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey)",
    "ALTER TABLE orders ADD FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey)",
    "ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey)"
]


# ==================== Database Functions ====================

def get_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        return conn
    except Exception as e:
        print(f"ERROR: Cannot connect to database: {e}")
        print("Make sure PostgreSQL is running and database 'tpch' exists.")
        sys.exit(1)


def drop_tables_if_exist():
    """Drop all TPC-H tables if they exist (clean start)"""
    conn = get_connection()
    cursor = conn.cursor()

    for table in reversed(TABLES):  # Reverse order to respect foreign keys
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            print(f"  Dropped table: {table}")
        except Exception as e:
            print(f"  Warning: Could not drop {table}: {e}")

    conn.commit()
    conn.close()
    print("All tables dropped.\n")


def create_tables():
    """Create all TPC-H tables"""
    conn = get_connection()
    cursor = conn.cursor()

    for table_name, create_sql in CREATE_TABLE_SQL.items():
        try:
            cursor.execute(create_sql)
            print(f"  Created table: {table_name}")
        except Exception as e:
            print(f"  ERROR creating {table_name}: {e}")

    conn.commit()
    conn.close()
    print("All tables created.\n")


def add_keys():
    """Add primary and foreign keys"""
    conn = get_connection()
    cursor = conn.cursor()

    print("Adding primary keys...")
    for sql in ADD_PRIMARY_KEYS:
        try:
            cursor.execute(sql)
            print(f"  {sql[:50]}... OK")
        except Exception as e:
            print(f"  Warning: {e}")

    print("\nAdding foreign keys...")
    for sql in ADD_FOREIGN_KEYS:
        try:
            cursor.execute(sql)
            print(f"  {sql[:50]}... OK")
        except Exception as e:
            print(f"  Warning: {e}")

    conn.commit()
    conn.close()
    print("\nAll keys added.\n")


def import_data():
    """Import CSV data into tables"""
    conn = get_connection()
    cursor = conn.cursor()

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found!")
        conn.close()
        sys.exit(1)

    for table in TABLES:
        csv_file = os.path.join(DATA_DIR, f"{table}.csv")

        if not os.path.exists(csv_file):
            print(f"  WARNING: {csv_file} not found, skipping {table}")
            continue

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # 不要跳过第一行！因为文件没有列名
                cursor.copy_from(f, table, sep='|', null='')
            print(f"  Imported: {table} from {csv_file}")
            conn.commit()
        except Exception as e:
            print(f"  ERROR importing {table}: {e}")
            conn.rollback()

    conn.close()
    print("\nData import completed.\n")

def verify_data():
    """Print row counts for verification"""
    conn = get_connection()
    cursor = conn.cursor()

    print("\n" + "=" * 50)
    print("Verification: Row counts")
    print("=" * 50)

    for table in TABLES:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:12s}: {count:6d} rows")
        except Exception as e:
            print(f"  {table:12s}: ERROR - {e}")

    conn.close()
    print("=" * 50 + "\n")


# ==================== Main ====================

def main():
    """Main execution"""
    print("=" * 60)
    print("TPC-H Database Setup")
    print("=" * 60)
    print(f"Database: {DB_NAME}")
    print(f"Data scale: {DATA_SCALE}")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60 + "\n")

    # Ask for confirmation
    response = input("This will drop existing tables and re-import data. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    print("\n[1/5] Dropping existing tables...")
    drop_tables_if_exist()

    print("[2/5] Creating tables...")
    create_tables()

    print("[3/5] Adding primary/foreign keys...")
    add_keys()

    print("[4/5] Importing data...")
    import_data()

    print("[5/5] Verifying data...")
    verify_data()

    print("Setup completed successfully!")


if __name__ == "__main__":
    main()
