import numpy as np
import pulp
import csv
from collections import Counter

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
                data.append(cust_map[row[0]])
    return data

def solve_lp(rows, tau):
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
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.value(prob.objective) or 0.0

def r2t_once(rows, epsilon, beta, GSQ):
    true_val = len(rows)
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
    data = load_data()
    print(f"Q3: {len(data)} rows")
    print(f"{'GSQ':<6} {'Error (%)':<12}")
    print("-" * 20)
    for gsq in gsq_list:
        errors = []
        for _ in range(50):
            errors.append(r2t_once(data, epsilon, beta, gsq))
        errors.sort()
        remove = int(50 * 0.2)
        avg_err = sum(errors[remove:-remove]) / len(errors[remove:-remove])
        print(f"{gsq:<6} {avg_err:.2f}")

if __name__ == "__main__":
    main()