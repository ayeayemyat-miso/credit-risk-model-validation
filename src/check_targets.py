"""
check_targets.py
----------------
Diagnostic script to verify the default target definition and
the last-payment-date proxy before finalising the PD target.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "loans_clean.parquet"


def main() -> None:
    df = pd.read_parquet(IN)
    co = df[df["loan_status"] == "Charged Off"].copy()

    print("=" * 70)
    print("CHARGED OFF LOANS: TERM DISTRIBUTION")
    print("=" * 70)
    print(co["term_months"].value_counts())
    print(f"\nTotal Charged Off: {len(co):,}")

    print("\n" + "=" * 70)
    print("months_to_last_pymnt BY TERM")
    print("=" * 70)
    print(co.groupby("term_months")["months_to_last_pymnt"].describe().round(1))

    print("\n" + "=" * 70)
    print("ANOMALY CHECK: m2lp > term_months")
    print("=" * 70)
    anomaly = co[co["months_to_last_pymnt"] > co["term_months"]]
    print(f"Count: {len(anomaly)}")
    if len(anomaly) > 0:
        print(anomaly[["id", "term_months", "months_to_last_pymnt", "loan_amnt"]].head(10))

    print("\n" + "=" * 70)
    print("QUANTILES OF months_to_last_pymnt (Charged Off)")
    print("=" * 70)
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(co["months_to_last_pymnt"].quantile(quantiles).round(1))

    print("\n" + "=" * 70)
    print("EDGE CASES: m2lp <= 0")
    print("=" * 70)
    print(f"  = 0: {(co['months_to_last_pymnt'] == 0).sum()}")
    print(f"  < 0: {(co['months_to_last_pymnt'] < 0).sum()}")
    print(f"  NaN: {co['months_to_last_pymnt'].isna().sum()}")

    print("\n" + "=" * 70)
    print("CUMULATIVE CAPTURE OF DEFAULTS BY WINDOW")
    print("=" * 70)
    for w in [6, 12, 18, 24, 30, 36, 48, 60]:
        n = (co["months_to_last_pymnt"] <= w).sum()
        print(f"  ≤ {w:>2}m : {n:>7,}  ({n/len(co)*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("HARD CHECK: ANY Charged Off WITH m2lp AFTER LOAN TERM")
    print("=" * 70)
    if len(anomaly) > 0:
        print(anomaly[["term_months", "months_to_last_pymnt"]]
              .value_counts()
              .head(20))


if __name__ == "__main__":
    main()