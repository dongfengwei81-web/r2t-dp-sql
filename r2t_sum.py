import psycopg2
import numpy as np
import pulp
import time


def get_db_connection():
    conn = psycopg2.connect(
        user="postgres",
        password="your password",
        host="localhost",
        port="5432",
        database="tpch"
    )
    return conn


def fetch_query_result(sql):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows


def solve_lp(data_rows, tau):
    """COUNT 模式的 LP（每行权重为1）"""
    if len(data_rows) == 0:
        return 0.0

    user_to_indices = {}
    for i, row in enumerate(data_rows):
        uid = row[0]
        if uid not in user_to_indices:
            user_to_indices[uid] = []
        user_to_indices[uid].append(i)

    prob = pulp.LpProblem("Truncation", pulp.LpMaximize)
    vars = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(len(data_rows))]
    prob += pulp.lpSum(vars)

    for uid, indices in user_to_indices.items():
        prob += pulp.lpSum(vars[i] for i in indices) <= tau

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    result = pulp.value(prob.objective)
    return result if result is not None else 0.0


def solve_lp_sum(data_rows, tau):
    """SUM 模式的 LP（每行权重为 weight）"""
    if len(data_rows) == 0:
        return 0.0

    user_to_indices = {}
    for i, row in enumerate(data_rows):
        uid = row[0]  # 用户ID
        if uid not in user_to_indices:
            user_to_indices[uid] = []
        user_to_indices[uid].append(i)

    prob = pulp.LpProblem("Truncation_SUM", pulp.LpMaximize)

    # 变量：每个变量可以取 0 到 weight 之间的值
    vars = []
    for i, row in enumerate(data_rows):
        weight = float(row[1])  # 第二列是权重值
        var = pulp.LpVariable(f"x_{i}", 0, weight, cat=pulp.LpContinuous)
        vars.append(var)

    # 目标：最大化保留的总权重
    prob += pulp.lpSum(vars)

    # 约束：每个用户的总权重 <= tau
    for uid, indices in user_to_indices.items():
        prob += pulp.lpSum(vars[i] for i in indices) <= tau

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    result = pulp.value(prob.objective)
    return result if result is not None else 0.0


def run_count_gsq(sql, query_name, GSQ, epsilon, beta, num_runs=100):
    """跑单个 GSQ 的 COUNT 实验（100 次）"""
    data = fetch_query_result(sql)
    real_count = len(data)

    if real_count == 0:
        print(f"  警告：查询结果为空")
        return None

    results = []
    log_gsq = np.log2(GSQ)

    for _ in range(num_runs):
        best = 0
        for j in range(1, int(log_gsq) + 1):
            tau = 2 ** j
            truncated = solve_lp(data, tau)
            noise = np.random.laplace(0, log_gsq * tau / epsilon)
            penalty = np.log(log_gsq / beta) * (tau / epsilon)
            noisy_result = truncated + noise - penalty
            best = max(best, noisy_result)

        error = abs(real_count - best) / real_count * 100
        results.append(error)

    # 去掉最好和最差各 20%
    num_remove = int(num_runs * 0.2)
    if num_remove > 0 and len(results) > 2 * num_remove:
        sorted_results = sorted(results)
        filtered = sorted_results[num_remove:-num_remove]
        avg_error = sum(filtered) / len(filtered)
    else:
        avg_error = sum(results) / len(results)

    print(f"  GSQ={GSQ} 完成，平均误差: {avg_error:.2f}%")
    return avg_error


def run_sum_gsq(sql, query_name, GSQ, epsilon, beta, num_runs=100):
    """跑单个 GSQ 的 SUM 实验（100 次）"""
    data = fetch_query_result(sql)
    # SUM 结果 = 所有权重的和
    real_result = sum(float(row[1]) for row in data)

    if real_result == 0:
        print(f"  警告：SUM 结果为空")
        return None

    results = []
    log_gsq = np.log2(GSQ)

    for _ in range(num_runs):
        best = 0
        for j in range(1, int(log_gsq) + 1):
            tau = 2 ** j
            truncated = solve_lp_sum(data, tau)
            noise = np.random.laplace(0, log_gsq * tau / epsilon)
            penalty = np.log(log_gsq / beta) * (tau / epsilon)
            noisy_result = truncated + noise - penalty
            best = max(best, noisy_result)

        error = abs(real_result - best) / real_result * 100
        results.append(error)

    num_remove = int(num_runs * 0.2)
    if num_remove > 0 and len(results) > 2 * num_remove:
        sorted_results = sorted(results)
        filtered = sorted_results[num_remove:-num_remove]
        avg_error = sum(filtered) / len(filtered)
    else:
        avg_error = sum(results) / len(results)

    print(f"  GSQ={GSQ} 完成，平均误差: {avg_error:.2f}%")
    return avg_error


def compare_gsq():
    """对比不同 GSQ 的效果"""

    EPSILON = 0.8
    BETA = 0.1
    NUM_RUNS = 100

    # ==================== 在这里选择查询类型 ====================
    QUERY_TYPE = "SUM"  # 可选: "COUNT_Q12", "COUNT_Q3", "COUNT_Q20", "SUM"
    # ============================================================

    if QUERY_TYPE == "COUNT_Q12":
        sql = """
            SELECT o_orderkey 
            FROM orders, lineitem 
            WHERE o_orderkey = l_orderkey
        """
        query_name = "Q12 (COUNT, 2表)"
        gsq_list = [2, 4, 8, 16, 32, 64, 128, 256]
        run_func = run_count_gsq

    elif QUERY_TYPE == "COUNT_Q3":
        sql = """
            SELECT customer.c_custkey
            FROM customer, orders, lineitem
            WHERE orders.O_CUSTKEY = customer.C_CUSTKEY
            AND lineitem.L_ORDERKEY = orders.O_ORDERKEY
        """
        query_name = "Q3 (COUNT, 3表)"
        gsq_list = [2, 4, 8, 16, 32, 64, 128, 256]
        run_func = run_count_gsq

    elif QUERY_TYPE == "COUNT_Q20":
        sql = """
            SELECT s_suppkey
            FROM supplier, partsupp, lineitem
            WHERE l_partkey = ps_partkey
            AND l_suppkey = ps_suppkey
        """
        query_name = "Q20 (COUNT, 4表)"
        gsq_list = [2, 4, 8, 16, 32, 64, 128, 256]
        run_func = run_count_gsq

    else:  # SUM
        # Q18: 每个客户的总订单数量
        sql = """
            SELECT c_custkey, l_quantity
            FROM customer, orders, lineitem
            WHERE c_custkey = o_custkey
            AND l_orderkey = o_orderkey
        """
        query_name = "Q18 (SUM)"
        # SUM 的真实值较大，GSQ 也要相应调大
        gsq_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        run_func = run_sum_gsq

    print("=" * 60)
    print(f"测试 {query_name}")
    print(f"参数: epsilon={EPSILON}, beta={BETA}")
    print(f"每个 GSQ 跑 {NUM_RUNS} 次")
    print("=" * 60)

    results = {}
    for gsq in gsq_list:
        avg_err = run_func(sql, query_name, gsq, EPSILON, BETA, NUM_RUNS)
        if avg_err is not None:
            results[gsq] = avg_err

    # 打印对比结果
    print("\n" + "=" * 60)
    print("GSQ 对比结果")
    print("=" * 60)
    print(f"{'GSQ':<10} {'平均误差 (%)':<15}")
    print("-" * 30)

    best_gsq = None
    best_error = float('inf')

    for gsq, err in results.items():
        print(f"{gsq:<10} {err:<15.2f}")
        if err < best_error:
            best_error = err
            best_gsq = gsq

    print("-" * 30)
    print(f"最优 GSQ: {best_gsq}, 对应误差: {best_error:.2f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    compare_gsq()
