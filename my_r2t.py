"""
R2T: Core Algorithm Implementation (Full TPC-H, 8 tables)
Author: Dong Fengwei
Date: 2026-05-16

This script implements the R2T (Race-to-the-Top) mechanism for
differentially private COUNT and SUM queries using TPC-H data.

Queries implemented:
- Q3: COUNT - number of orders per customer (customer -> orders -> lineitem)
- Q12: COUNT - number of lineitems per order (orders -> lineitem)
- Q18: SUM - total quantity per customer (customer -> orders -> lineitem)
- Q20: COUNT - number of lineitems per supplier (supplier -> partsupp -> lineitem)

Data source: TPC-H scale factor 0.125 (8 CSV files)
- customer.csv, orders.csv, lineitem.csv
- supplier.csv, partsupp.csv, part.csv
- nation.csv, region.csv
"""

import os
import tempfile
os.environ['TMPDIR'] = 'F:/pulp_temp'
tempfile.tempdir = 'F:/pulp_temp'
os.makedirs('F:/pulp_temp', exist_ok=True)
import numpy as np
import pulp
import csv
from collections import Counter, defaultdict
# ============================================================
# Configuration
# ============================================================

DATA_DIR = "F:/IP/IP-21273301/Data/_0.125"  # Please modify to your data path

# TPC-H CSV file column indices (pipe-delimited, no header)
# Reference: https://github.com/electrum/tpch-dbgen
COL_ORDER = {
    'orders': {
        'o_orderkey': 0,
        'o_custkey': 1,
        'o_orderstatus': 2,
        'o_totalprice': 3,
        'o_orderdate': 4,
        'o_orderpriority': 5,
        'o_clerk': 6,
        'o_shippriority': 7,
        'o_comment': 8
    },
    'lineitem': {
        'l_orderkey': 0,
        'l_partkey': 1,
        'l_suppkey': 2,
        'l_linenumber': 3,
        'l_quantity': 4,
        'l_extendedprice': 5,
        'l_discount': 6,
        'l_tax': 7,
        'l_returnflag': 8,
        'l_linestatus': 9,
        'l_shipdate': 10,
        'l_commitdate': 11,
        'l_receiptdate': 12,
        'l_shipinstruct': 13,
        'l_shipmode': 14,
        'l_comment': 15
    },
    'customer': {
        'c_custkey': 0,
        'c_name': 1,
        'c_address': 2,
        'c_nationkey': 3,
        'c_phone': 4,
        'c_acctbal': 5,
        'c_mktsegment': 6,
        'c_comment': 7
    },
    'supplier': {
        's_suppkey': 0,
        's_name': 1,
        's_address': 2,
        's_nationkey': 3,
        's_phone': 4,
        's_acctbal': 5,
        's_comment': 6
    },
    'partsupp': {
        'ps_partkey': 0,
        'ps_suppkey': 1,
        'ps_availqty': 2,
        'ps_supplycost': 3,
        'ps_comment': 4
    }
}


# ============================================================
# Data Loading Functions (using 8 CSV files)
# ============================================================

def load_csv(file_name, delimiter='|'):
    """Load CSV file, return list of rows"""
    file_path = os.path.join(DATA_DIR, file_name)
    rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for row in csv.reader(f, delimiter=delimiter):
            if row and len(row) > 1:  # Skip empty rows
                rows.append(row)
    return rows


def load_q3_data():
    """
    Q3: Number of orders per customer
    Table join: customer -> orders -> lineitem
    Returns: [c_custkey, ...] one record per lineitem
    """
    orders = load_csv('orders.csv')
    order_to_cust = {row[COL_ORDER['orders']['o_orderkey']]: row[COL_ORDER['orders']['o_custkey']]
                     for row in orders}

    lineitem = load_csv('lineitem.csv')
    data = []
    for row in lineitem:
        orderkey = row[COL_ORDER['lineitem']['l_orderkey']]
        if orderkey in order_to_cust:
            custkey = order_to_cust[orderkey]
            data.append(custkey)

    return data


def load_q12_data():
    """
    Q12: Number of lineitems per order
    Table join: orders -> lineitem
    Returns: [o_orderkey, ...] one record per lineitem
    """
    orders = load_csv('orders.csv')
    valid_orderkeys = {row[COL_ORDER['orders']['o_orderkey']] for row in orders}

    lineitem = load_csv('lineitem.csv')
    data = []
    for row in lineitem:
        orderkey = row[COL_ORDER['lineitem']['l_orderkey']]
        if orderkey in valid_orderkeys:
            data.append(orderkey)

    return data


def load_q18_data():
    """
    Q18: Total order quantity per customer (SUM aggregation)
    Table join: customer -> orders -> lineitem
    Returns: [(c_custkey, l_quantity), ...] each lineitem with weight
    """
    orders = load_csv('orders.csv')
    order_to_cust = {row[COL_ORDER['orders']['o_orderkey']]: row[COL_ORDER['orders']['o_custkey']]
                     for row in orders}

    lineitem = load_csv('lineitem.csv')
    data = []
    for row in lineitem:
        orderkey = row[COL_ORDER['lineitem']['l_orderkey']]
        if orderkey in order_to_cust:
            custkey = order_to_cust[orderkey]
            quantity = float(row[COL_ORDER['lineitem']['l_quantity']])
            data.append((custkey, quantity))

    return data


def load_q20_data():
    """
    Q20: Number of lineitems per supplier
    Table join: supplier -> partsupp -> lineitem
    匹配条件: (l_partkey, l_suppkey) = (ps_partkey, ps_suppkey)
    Returns: [s_suppkey, ...] one record per matched lineitem
    """
    # Load partsupp: 构建 (partkey, suppkey) 集合
    partsupp = load_csv('partsupp.csv')
    part_supp_pairs = set()
    for row in partsupp:
        partkey = row[COL_ORDER['partsupp']['ps_partkey']]
        suppkey = row[COL_ORDER['partsupp']['ps_suppkey']]
        part_supp_pairs.add((partkey, suppkey))

    # Load supplier: 构建有效 suppkey 集合
    suppliers = load_csv('supplier.csv')
    valid_suppkeys = {row[COL_ORDER['supplier']['s_suppkey']] for row in suppliers}

    # Load lineitem, 匹配 (l_partkey, l_suppkey) 到 (ps_partkey, ps_suppkey)
    data = []
    lineitem = load_csv('lineitem.csv')
    for row in lineitem:
        partkey = row[COL_ORDER['lineitem']['l_partkey']]
        suppkey = row[COL_ORDER['lineitem']['l_suppkey']]
        if (partkey, suppkey) in part_supp_pairs and suppkey in valid_suppkeys:
            data.append(suppkey)

    return data


# ============================================================
# Data Integrity Verification
# ============================================================

def verify_data():
    """Verify that all 8 CSV files exist and print basic information"""
    files = ['customer.csv', 'orders.csv', 'lineitem.csv',
             'supplier.csv', 'partsupp.csv', 'part.csv',
             'nation.csv', 'region.csv']

    print("=" * 60)
    print("Data File Verification")
    print("=" * 60)

    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                line_count = sum(1 for _ in fp)
            print(f"[OK] {f}: {line_count:,} rows")
        else:
            print(f"[MISSING] {f}: does not exist")

    print("=" * 60)


# ============================================================
# LP Solver
# ============================================================

def build_lp_count(rows, tau):
    """LP for COUNT: maximize rows kept, each user <= tau"""
    if len(rows) == 0:
        return 0.0

    user_counts = Counter(rows)
    prob = pulp.LpProblem("", pulp.LpMaximize)
    vars = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(len(rows))]
    prob += pulp.lpSum(vars)

    idx = 0
    for cnt in user_counts.values():
        prob += pulp.lpSum(vars[idx:idx+cnt]) <= tau
        idx += cnt

    prob.solve(pulp.PULP_CBC_CMD(msg=False, keepFiles=0))
    return pulp.value(prob.objective) or 0.0

def build_lp_sum(rows, tau):
    """LP for SUM: maximize total weight kept, each user's weight <= tau"""
    if len(rows) == 0:
        return 0.0

    user_weights = defaultdict(list)
    for i, (uid, w) in enumerate(rows):
        user_weights[uid].append((i, w))

    prob = pulp.LpProblem("", pulp.LpMaximize)
    vars = [pulp.LpVariable(f"x_{i}", 0, w) for i, (_, w) in enumerate(rows)]
    prob += pulp.lpSum(vars)

    for items in user_weights.values():
        prob += pulp.lpSum(vars[i] for i, _ in items) <= tau

    # prob.solve(pulp.PULP_CBC_CMD(msg=False, keepFiles=0, tmpDir='F:/pulp_temp'))
    prob.solve(pulp.PULP_CBC_CMD(msg=False, keepFiles=0))
    return pulp.value(prob.objective) or 0.0


# ============================================================
# R2T Algorithm
# ============================================================

def r2t_count(data, epsilon, beta, GSQ, query_name=""):
    """R2T for COUNT queries with MAX operation (single run)"""
    true_val = len(data)
    if true_val == 0:
        print(f"Warning: {query_name} data is empty")
        return 0, 0.0

    best = 0
    log_gsq = np.log2(GSQ)

    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = build_lp_count(data, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        # 修正：惩罚项加上 log_gsq 因子，与论文公式 (7) 一致
        penalty = log_gsq * np.log(log_gsq / beta) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)

    error = abs(true_val - best) / true_val * 100
    print(f"True value: {true_val:,.0f}, R2T output: {best:.0f}, Error: {error:.2f}%")
    return best, error


def r2t_sum(data, epsilon, beta, GSQ, query_name=""):
    """R2T for SUM queries with MAX operation (single run)"""
    true_val = sum(w for _, w in data)
    if true_val == 0:
        print(f"Warning: {query_name} data is empty")
        return 0, 0.0

    best = 0
    log_gsq = np.log2(GSQ)

    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = build_lp_sum(data, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        # 修正：惩罚项加上 log_gsq 因子，与论文公式 (7) 一致
        penalty = log_gsq * np.log(log_gsq / beta) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)

    error = abs(true_val - best) / true_val * 100
    print(f"True value: {true_val:,.0f}, R2T output: {best:.0f}, Error: {error:.2f}%")
    return best, error


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":
    EPS, BETA = 0.8, 0.1

    # Verify data files
    verify_data()

    print("\n" + "=" * 60)
    print("R2T Demo - Single Run Results (Full TPC-H, 8 tables)")
    print("=" * 60)

    # Q12: 2-table join (orders + lineitem)
    print("\n[Q12] COUNT (orders -> lineitem, GSQ=16):")
    data_q12 = load_q12_data()
    r2t_count(data_q12, EPS, BETA, 16, "Q12")

    # Q3: 3-table join (customer -> orders -> lineitem)
    print("\n[Q3] COUNT (customer -> orders -> lineitem, GSQ=64):")
    data_q3 = load_q3_data()
    r2t_count(data_q3, EPS, BETA, 64, "Q3")

    # Q20: 3-table join (supplier -> partsupp -> lineitem)
    print("\n[Q20] COUNT (supplier -> partsupp -> lineitem, GSQ=128):")
    data_q20 = load_q20_data()
    r2t_count(data_q20, EPS, BETA, 128, "Q20")

    # Q18: SUM aggregation (customer -> orders -> lineitem)
    print("\n[Q18] SUM (customer -> orders -> lineitem, GSQ=1024):")
    data_q18 = load_q18_data()
    r2t_sum(data_q18, EPS, BETA, 1024, "Q18")

    print("\n" + "=" * 60)
