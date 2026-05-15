"""
R2T: Core Algorithm Implementation
Author: Dong Fengwei
Date: 2026-05-14

This script implements the R2T (Race-to-the-Top) mechanism for
differentially private COUNT and SUM queries.

Key functions:
- build_lp_count(): LP solver for COUNT queries
- build_lp_sum(): LP solver for SUM queries
- r2t_count(): R2T algorithm for COUNT
- r2t_sum(): R2T algorithm for SUM
- MAX operation: max(best, noisy_result)
"""

import psycopg2
import numpy as np
import pulp


def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        user="postgres",
        password="174110",
        host="localhost",
        port="5432",
        database="tpch"
    )


def execute_sql(sql):
    """Execute SQL query and return results"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows


def build_lp_count(rows, tau):
    """LP for COUNT: maximize rows kept, each user <= tau"""
    if len(rows) == 0:
        return 0.0

    # Group by user ID
    user_groups = {}
    for idx, row in enumerate(rows):
        uid = row[0]
        if uid not in user_groups:
            user_groups[uid] = []
        user_groups[uid].append(idx)

    prob = pulp.LpProblem("R2T_COUNT", pulp.LpMaximize)
    vars = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(len(rows))]
    prob += pulp.lpSum(vars)

    for indices in user_groups.values():
        prob += pulp.lpSum(vars[i] for i in indices) <= tau

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    result = pulp.value(prob.objective)
    return result if result is not None else 0.0


def build_lp_sum(rows, tau):
    """LP for SUM: maximize total weight kept, each user's weight <= tau"""
    if len(rows) == 0:
        return 0.0

    user_groups = {}
    for idx, row in enumerate(rows):
        uid = row[0]
        if uid not in user_groups:
            user_groups[uid] = []
        user_groups[uid].append(idx)

    prob = pulp.LpProblem("R2T_SUM", pulp.LpMaximize)
    vars = []
    for idx, row in enumerate(rows):
        weight = float(row[1])
        vars.append(pulp.LpVariable(f"x_{idx}", 0, weight))
    prob += pulp.lpSum(vars)

    for indices in user_groups.values():
        prob += pulp.lpSum(vars[i] for i in indices) <= tau

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    result = pulp.value(prob.objective)
    return result if result is not None else 0.0


def r2t_count(sql, epsilon, beta, GSQ):
    """R2T for COUNT queries with MAX operation"""
    rows = execute_sql(sql)
    true_val = len(rows)

    if true_val == 0:
        return 0, 0.0

    best = 0
    log_gsq = np.log2(GSQ)

    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = build_lp_count(rows, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        penalty = np.log(log_gsq / 0.1) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)  # MAX operation

    error = abs(true_val - best) / true_val * 100
    print(f"True: {true_val}, R2T: {best:.2f}, Error: {error:.2f}%")
    return best, error


def r2t_sum(sql, epsilon, beta, GSQ):
    """R2T for SUM queries with MAX operation"""
    rows = execute_sql(sql)
    true_val = sum(float(row[1]) for row in rows)

    if true_val == 0:
        return 0, 0.0

    best = 0
    log_gsq = np.log2(GSQ)

    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = build_lp_sum(rows, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        penalty = np.log(log_gsq / 0.1) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)  # MAX operation

    error = abs(true_val - best) / true_val * 100
    print(f"True: {true_val:.2f}, R2T: {best:.2f}, Error: {error:.2f}%")
    return best, error


# SQL queries
SQL_Q12 = "SELECT o_orderkey FROM orders, lineitem WHERE o_orderkey = l_orderkey"
SQL_Q3 = """
    SELECT customer.c_custkey
    FROM customer, orders, lineitem
    WHERE orders.O_CUSTKEY = customer.C_CUSTKEY
    AND lineitem.L_ORDERKEY = orders.O_ORDERKEY
"""
SQL_Q20 = """
    SELECT s_suppkey
    FROM supplier, partsupp, lineitem
    WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
"""
SQL_Q18 = """
    SELECT c_custkey, l_quantity
    FROM customer, orders, lineitem
    WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
"""


if __name__ == "__main__":
    EPS, BETA = 0.8, 0.1
    print("Q12 (COUNT, GSQ=8):")
    r2t_count(SQL_Q12, EPS, BETA, 8)
    print("\nQ3 (COUNT, GSQ=64):")
    r2t_count(SQL_Q3, EPS, BETA, 64)
    print("\nQ20 (COUNT, GSQ=128):")
    r2t_count(SQL_Q20, EPS, BETA, 128)
    print("\nQ18 (SUM, GSQ=1024):")
    r2t_sum(SQL_Q18, EPS, BETA, 1024)