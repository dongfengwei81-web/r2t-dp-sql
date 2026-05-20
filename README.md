# R2T: Instance-Optimal Truncation for Differentially Private Query Evaluation

A Python implementation of the R2T mechanism (Dong et al., SIGMOD 2022) for answering SPJA queries under differential privacy with foreign key constraints.

## Quick Start

```bash
pip install -r requirements.txt
python my_r2t.py
Data
TPC-H scale factor 0.125, generated via tpchgen-cli:

Table	Rows
lineitem	750,594
orders	187,500
customer	18,750
partsupp	100,000
supplier	1,250
part	25,000
Usage
bash
python run_q12.py   # COUNT, 2-table join (50 runs)
python run_q3.py    # COUNT, 3-table join (50 runs)
python run_q20.py   # COUNT, 4-table join (50 runs)
python run_q18.py   # SUM aggregation (30 runs)
Parameters
Parameter	Value
ε	0.8
β	0.1
Repetitions	50 (COUNT), 30 (SUM)
Outlier trimming	20%
Results
TPC-H SF=0.125, 750,594 lineitems:

Query	Type	Optimal GSQ	Error
Q12	COUNT (orders→lineitem)	8	0.013%
Q3	COUNT (customer→orders→lineitem)	128	0.73%
Q20	COUNT (supplier→partsupp→lineitem)	1024	8.28%
Q18	SUM (customer→orders→lineitem)	4096	1.50%
Algorithm
For τ = 2, 4, 8, …, GSQ:

text
Q̃(τ) = Q_truncated(τ) + Lap(log₂(GSQ)·τ/ε) − log₂(GSQ)·ln(log₂(GSQ)/β)·τ/ε
Final output: max{ Q̃(τ), Q(0) }

File Structure
text
├── my_r2t.py          # Single-run demo
├── run_q3.py          # Q3 batch test
├── run_q12.py         # Q12 batch test
├── run_q20.py         # Q20 batch test
├── run_q18.py         # Q18 batch test
├── q*_results.csv     # Experimental data
└── report/            # Project report (PDF)
