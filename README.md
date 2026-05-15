+# R2T: Instance-Optimal Truncation for Differentially Private Query Evaluation

> A complete implementation of the R2T (Race-to-the-Top) mechanism for answering SPJA queries under differential privacy, as proposed by Dong et al. at SIGMOD 2022.


## Overview

This project implements the R2T mechanism for differentially private query evaluation on relational databases with foreign key constraints. The mechanism addresses the fundamental challenge of unbounded global sensitivity by:

- LP-Based Truncation: Linear programming to maximize retained tuples under per-user caps
- Adaptive Threshold Selection: Testing geometrically increasing thresholds (2, 4, 8, ..., GSQ)
- MAX Operation: Returning the maximum noisy estimate to guarantee monotonic improvement

The implementation supports both COUNT and SUM aggregations on TPC-H benchmark queries.

## Environment

- OS: Windows 11 / Linux
- Database: PostgreSQL 17 (optional)
- Python: 3.11+
- LP Solver: PuLP (CBC solver)

## Installation

### 1. Clone or download the code

git clone https://github.com/fdongab/r2t-dp-sql.git
cd r2t-dp-sql

### 2. Install dependencies

pip install -r requirements.txt

### 3. Database setup (optional)

If using PostgreSQL instead of CSV files:

CREATE DATABASE tpch;
-- Import TPC-H data at scale factor 0.125

Update database credentials in my_r2t.py or r2t_sum.py:

conn = psycopg2.connect(
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432",
    database="tpch"
)

### 4. CSV data directory

If using CSV files, ensure the directory structure:

F:/IP/IP-21273301/Data/_0.125/
|-- orders.csv
|-- lineitem.csv
|-- customer.csv
|-- supplier.csv
|-- partsupp.csv
|-- part.csv

CSV format: pipe-delimited (|) with no header row, matching TPC-H output.

## File Structure

r2t-dp-sql/
|-- my_r2t.py              # Main script (PostgreSQL, single run per GSQ)
|-- r2t_sum.py             # Comprehensive SUM/COUNT tester (PostgreSQL, 100 runs)
|-- run_q3.py              # Q3 COUNT test with CSV (50 runs)
|-- run_q12.py             # Q12 COUNT test with CSV (50 runs)
|-- run_q20.py             # Q20 COUNT test with CSV (50 runs)
|-- run_q18.py             # Q18 SUM test with CSV (30 runs)
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
|-- report/                # Project report (PDF)

## Usage

### Quick Start (Single Run)

Test all four queries with optimal GSQ values:

python my_r2t.py

Output will show true values and R2T estimates for Q12, Q3, Q20, Q18.

### Comprehensive Testing (Multiple GSQ Values)

To test across different GSQ values with statistical significance:

python r2t_sum.py

Before running, set QUERY_TYPE at the top of the file:

QUERY_TYPE = "COUNT_Q12"   # Options: COUNT_Q12, COUNT_Q3, COUNT_Q20, SUM

### Individual Query Tests (CSV-based)

python run_q3.py    # Q3 COUNT, tests GSQ=2,4,8,16,32,64,128,256
python run_q12.py   # Q12 COUNT, tests GSQ=2,4,8,16,32,64,128,256
python run_q20.py   # Q20 COUNT, tests GSQ=2,4,8,16,32,64,128,256
python run_q18.py   # Q18 SUM, tests GSQ=8,16,32,64,128,512,1024,2048,4096

### Modifying GSQ Test Range

Edit the gsq_list variable in each script:

# Example: test more GSQ values for Q12
gsq_list = [2, 4, 8, 16, 32, 64, 128, 256]

### Changing Number of Repetitions

Edit the num_runs or range parameter:

# For 100 repetitions:
for _ in range(100):
    errors.append(r2t_once(...))

## Core Algorithm

### COUNT Query LP Formulation

maximize    sum u_k
subject to  sum_{k in C_j} u_k <= tau,  for all j
            0 <= u_k <= 1,              for all k

### SUM Query LP Formulation

maximize    sum u_k
subject to  sum_{k in C_j} u_k <= tau,  for all j
            0 <= u_k <= psi(q_k),       for all k

### R2T Algorithm

For tau = 2, 4, 8, ..., GSQ:

Q_tilde(tau) = Q_truncated(tau) + Lap(log(GSQ)*tau/epsilon) - log(GSQ)*ln(log(GSQ)/beta)*(tau/epsilon)

Final output: max{ Q_tilde(tau), Q(0) }

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| epsilon   | 0.8   | Privacy budget |
| beta      | 0.1   | Failure probability |
| GSQ       | Varies| Global sensitivity bound |
| Repetitions| 50 (COUNT), 30 (SUM) | Statistical significance |
| Outlier removal | 20% | Remove best/worst 20% |

## Experimental Results

Optimal parameters found on TPC-H SF=0.125 (~125MB):

| Query | Type | Tables | Optimal GSQ | Minimum Error |
|-------|------|--------|-------------|---------------|
| Q12   | COUNT | orders, lineitem | 8 | 0.04% |
| Q3    | COUNT | customer, orders, lineitem | 64 | 0.31% |
| Q20   | COUNT | supplier, partsupp, lineitem | 128 | 1.87% |
| Q18   | SUM   | customer, orders, lineitem | 1024 | 0.44% |

## Key Implementation Details

### Laplace Noise Generation

noise = np.random.laplace(0, log_gsq * tau / epsilon)

### MAX Operation (Race-to-the-Top)

best = max(best, noisy_result)

### Outlier Removal for Averaging

errors.sort()
remove = int(len(errors) * 0.2)
avg_error = sum(errors[remove:-remove]) / len(errors[remove:-remove])

## Limitations and Future Work

Current limitations:
- No self-join validation (experiments focus on PK-FK joins between distinct tables)
- Sequential LP solving (parallel execution not implemented)
- Manual GSQ selection required
- No GROUP BY support

Future directions:
- Validate on self-join queries (e.g., triangle counting in graphs)
- Implement dual LP early-stop optimization
- Automatic GSQ estimation from data statistics
- GROUP BY query support
- Test with different epsilon values

## References

- Dong, W., Fang, J., Yi, K., Tao, Y., & Machanavajjhala, A. (2022). R2T: Instance-optimal Truncation for Differentially Private Query Evaluation with Foreign Keys. SIGMOD 2022.
- Kotsogiannis, I., et al. (2019). PrivateSQL: A differentially private SQL query engine. VLDB 2019.

## License

This project is for academic purposes as part of course requirements.

## Contact

GitHub: https://github.com/fdongab/r2t-dp-sql