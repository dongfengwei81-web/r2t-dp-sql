import numpy as np
import pulp
import csv
from collections import Counter, defaultdict

DATA_DIR = "F:/IP/IP-21273301/Data/_0.125"

# 定义列索引
COL_IDX = {
    'lineitem': {
        'l_orderkey': 0,
        'l_partkey': 1,
        'l_suppkey': 2,
        'l_quantity': 4,
    },
    'partsupp': {
        'ps_partkey': 0,
        'ps_suppkey': 1,
    },
    'supplier': {
        's_suppkey': 0,
    }
}


def load_q20_data():
    """
    Q20: 加载每个 lineitem 对应的 supplier
    返回: list of suppkey (每个 lineitem 一条记录)
    """
    # 构建 (partkey, suppkey) 对集合
    partsupp_pairs = set()
    with open(f"{DATA_DIR}/partsupp.csv", 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row and len(row) > 1:
                partkey = row[COL_IDX['partsupp']['ps_partkey']]
                suppkey = row[COL_IDX['partsupp']['ps_suppkey']]
                partsupp_pairs.add((partkey, suppkey))

    # 验证 supplier 存在性（可选）
    valid_suppkeys = set()
    with open(f"{DATA_DIR}/supplier.csv", 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row:
                valid_suppkeys.add(row[COL_IDX['supplier']['s_suppkey']])

    # 匹配 lineitem
    data = []
    with open(f"{DATA_DIR}/lineitem.csv", 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row and len(row) > 2:
                partkey = row[COL_IDX['lineitem']['l_partkey']]
                suppkey = row[COL_IDX['lineitem']['l_suppkey']]

                # 同时匹配 partkey 和 suppkey
                if (partkey, suppkey) in partsupp_pairs and suppkey in valid_suppkeys:
                    data.append(suppkey)

    print(f"Q20 数据加载完成: {len(data):,} 条记录")
    return data


def solve_lp(rows, tau):
    """LP solver for COUNT queries"""
    if len(rows) == 0:
        return 0.0

    user_counts = Counter(rows)
    prob = pulp.LpProblem("", pulp.LpMaximize)
    vars = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(len(rows))]
    prob += pulp.lpSum(vars)

    idx = 0
    for cnt in user_counts.values():
        prob += pulp.lpSum(vars[idx:idx + cnt]) <= tau
        idx += cnt

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(prob.objective) or 0.0


def r2t_once(rows, epsilon, beta, GSQ):
    """R2T single run"""
    true_val = len(rows)
    if true_val == 0:
        return 0.0

    best = 0
    log_gsq = np.log2(GSQ)

    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = solve_lp(rows, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        penalty = np.log(log_gsq / beta) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)

    return abs(true_val - best) / true_val * 100


def main():
    epsilon, beta = 0.8, 0.1
    gsq_list = [2, 4, 8, 16, 32, 64, 128, 256]

    # 加载数据
    data = load_q20_data()

    print("\n" + "=" * 50)
    print(f"R2T 实验: Q20 (COUNT Query)")
    print(f"ε = {epsilon}, β = {beta}")
    print(f"数据规模: {len(data):,} 条记录")
    print("=" * 50)
    print(f"{'GSQ':<8} {'Error (%)':<12} {'Std Dev':<12}")
    print("-" * 35)

    results = []
    for gsq in gsq_list:
        errors = []
        for _ in range(50):
            errors.append(r2t_once(data, epsilon, beta, gsq))

        # 计算均值和标准差（去掉20%的极端值）
        errors_sorted = sorted(errors)
        remove = int(len(errors_sorted) * 0.2)
        trimmed_errors = errors_sorted[remove:-remove] if remove > 0 else errors_sorted
        mean_err = np.mean(trimmed_errors)
        std_err = np.std(trimmed_errors)

        print(f"{gsq:<8} {mean_err:.4f}     {std_err:.4f}")
        results.append({"GSQ": gsq, "mean_error": mean_err, "std_error": std_err})

    # 保存结果
    with open("q20_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["GSQ", "mean_error", "std_error"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print("结果已保存到 q20_results.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
