# Credit Risk Model Validation — LendingClub Portfolio

An independent Python project demonstrating credit risk modelling, model validation, illustrative Expected Credit Loss (ECL), hypothetical Risk-Weighted Assets (RWA), and stress testing using historical LendingClub loan data.

**Tools:** Python, pandas, scikit-learn, Streamlit

## Project Overview

| Module | Method |
|---|---|
| PD | Discrete-time logistic hazard model |
| LGD | Historical recovery-based training-mean baseline |
| EAD | Contractual amortisation schedules |
| Staging | Synthetic IFRS 9-inspired Stage 1/2 classification |
| ECL | Monthly marginal PD × LGD × EAD × discount factor |
| RWA | Hypothetical exposure-based risk weights |
| Validation | Out-of-time performance and calculation checks |
| Stress Testing | Hypothetical PD and LGD sensitivity scenarios |

## Dataset

- Source: LendingClub historical loan data (2007–2018Q4)
- Filtered population: 444,919 loans
- Modelling sample: **66,688 loans**
- Training: 2011–2013 vintages
- Out-of-time testing: 2014 vintage
- Current-status subset: **1,789 loans**

## Key Results

| Metric | Result |
|---|---:|
| PD ROC-AUC | 0.6871 |
| PD Gini | 0.3741 |
| PD Brier Score | 0.01947 |
| LGD MAE | 0.1106 |
| LGD RMSE | 0.1752 |
| Current selected synthetic ECL | 1,285,357 |
| Illustrative RWA (100% weight) | 4,413,626 |
| Combined adverse stress ECL increase | +67.57% |

PD validation uses 30,827 eligible out-of-time test loans. Monetary amounts are in source-currency units.

## Interactive Dashboard

A Streamlit dashboard presents:

- Portfolio overview and credit risk metrics
- PD/LGD model validation and explanations
- Illustrative ECL and RWA calculations
- Hypothetical stress-testing results

**Run locally:**

\`\`\`powershell
streamlit run app.py
\`\`\`

## Limitations

This is an **educational portfolio project**, not a production credit risk system or a regulatory IFRS 9/Basel capital model.

Key limitations include synthetic staging, historical recovery-based LGD, contractual EAD assumptions, incomplete default follow-up, non-EIR discounting, and hypothetical risk weights and stress scenarios.