import os
import tempfile
os.environ['TMPDIR'] = 'F:/pulp_temp'
tempfile.tempdir = 'F:/pulp_temp'
os.makedirs('F:/pulp_temp', exist_ok=True)

import numpy as np
import pulp
import csv
from collections import defaultdict

DATA_DIR = "F:/IP/IP-21273301/Data/_0.125"


def load_data():
    cust_map = {}
    with open(f"{DATA_DIR}/orders.csv", 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row:
                cust_map[row[0]] = row[1]
    data = []
    with open(f"{DATA_DIR}/lineitem.csv", 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row and row[0] in cust_map:
                data.append((cust_map[row[0]], float(row[4])))
    return data


def solve_lp(rows, tau):
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
    prob.solve(pulp.PULP_CBC_CMD(msg=False, keepFiles=0))
    return pulp.value(prob.objective) or 0.0


def r2t_once(rows, epsilon, beta, GSQ):
    true_val = sum(w for _, w in rows)
    best = 0
    log_gsq = np.log2(GSQ)
    for j in range(1, int(log_gsq) + 1):
        tau = 2 ** j
        truncated = solve_lp(rows, tau)
        noise = np.random.laplace(0, log_gsq * tau / epsilon)
        penalty = log_gsq * np.log(log_gsq / beta) * (tau / epsilon)
        noisy = truncated + noise - penalty
        best = max(best, noisy)
    return abs(true_val - best) / true_val * 100


def main():
    epsilon, beta = 0.8, 0.1
    gsq_list = [256,512,1024,2048,4096,8129, 16384]
    data = load_data()
    true_val = sum(w for _, w in data)
    print(f"Q18 (SUM): {len(data)} rows")
    print(f"True value: {true_val:.2f}")
    print(f"{'GSQ':<8} {'Error (%)':<12} {'Std Dev':<12}")
    print("-" * 35)

    results = []

    for gsq in gsq_list:
        errors = []
        for _ in range(30):
            errors.append(r2t_once(data, epsilon, beta, gsq))
        errors.sort()
        remove = int(30 * 0.2)
        trimmed_errors = errors[remove:-remove]
        avg_err = sum(trimmed_errors) / len(trimmed_errors)
        std_err = np.std(trimmed_errors)
        print(f"{gsq:<8} {avg_err:.2f}        {std_err:.2f}")
        results.append({"GSQ": gsq, "avg_error": avg_err, "std_dev": std_err})

    with open("q18_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["GSQ", "avg_error", "std_dev"])
        writer.writeheader()
        writer.writerows(results)

    print("\n结果已保存到 q18_results.csv")


if __name__ == "__main__":
    main()
