"""
inspect_data.py
---------------
Phase 0 — Data feasibility assessment for the LendingClub dataset.
"""

import pandas as pd
from pathlib import Path

# ---- Path setup ----
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "loan_data.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def main() -> None:
    section("LOADING DATA")
    df = pd.read_csv(
        RAW,
        low_memory=False,
        usecols=[
            "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
            "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
            "verification_status", "issue_d", "loan_status", "purpose",
            "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
            "fico_range_low", "fico_range_high", "inq_last_6mths",
            "mths_since_last_delinq", "mths_since_last_record",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "application_type", "mort_acc", "pub_rec_bankruptcies", "tax_liens",
            "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
            "total_il_high_credit_limit", "avg_cur_bal", "bc_open_to_buy", "bc_util",
            "acc_open_past_24mths", "num_actv_bc_tl", "num_actv_rev_tl",
            "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
            "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
            "pct_tl_nvr_dlq", "percent_bc_gt_75",
            "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
            "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
            "mths_since_recent_bc", "mths_since_recent_inq",
            "mths_since_recent_revol_delinq",
            "total_rev_hi_lim", "all_util", "il_util", "max_bal_bc",
            "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
            "mths_since_rcnt_il", "total_bal_il",
            "open_rv_12m", "open_rv_24m", "inq_fi", "total_cu_tl", "inq_last_12m",
            "tot_coll_amt", "tot_cur_bal", "acc_now_delinq", "delinq_amnt",
            "num_accts_ever_120_pd", "num_tl_120dpd_2m", "num_tl_30dpd",
            "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "chargeoff_within_12_mths",
            "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
            "collections_12_mths_ex_med",
            "last_pymnt_d", "next_pymnt_d",
            "recoveries", "collection_recovery_fee",
            "total_rec_prncp", "out_prncp", "total_pymnt",
            "hardship_flag", "hardship_status",
            "hardship_start_date", "hardship_end_date",
            "hardship_dpd", "hardship_loan_status",
            "debt_settlement_flag", "settlement_date",
            "settlement_amount", "settlement_percentage",
            "annual_inc_joint", "dti_joint", "verification_status_joint",
            "revol_bal_joint",
            "sec_app_fico_range_low", "sec_app_fico_range_high",
        ],
    )

    # Parse dates
    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_d_parsed"].dt.year

    # Sample filter: keep 2011-2014 vintages and core loan statuses
    df = df[df["issue_year"].between(2011, 2014)]
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Current"])]

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"File:  {RAW}")

    section("1. COLUMN INVENTORY")
    print(f"Total columns: {len(df.columns)}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")

    section("2. LOAN STATUS DISTRIBUTION")
    print(df["loan_status"].value_counts(dropna=False))
    print("\nAs % of total:")
    print((df["loan_status"].value_counts(normalize=True, dropna=False) * 100).round(2))

    section("3. ORIGINATION VINTAGES")
    print("Loans per issue year:")
    print(df["issue_year"].value_counts().sort_index())
    print(f"\nUnparseable issue_d: {df['issue_d_parsed'].isna().sum():,}")

    section("4. TIMING COLUMNS")
    for col in ["issue_d", "last_pymnt_d", "next_pymnt_d", "earliest_cr_line"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  {col:25s} non-null: {nn:>10,}  ({nn/len(df)*100:5.2f}%)")
        else:
            print(f"  {col:25s} NOT PRESENT")

    section("4b. TIMING BY LOAN STATUS")
    if "last_pymnt_d" in df.columns:
        print("last_pymnt_d non-null rate by loan_status:")
        print(df.groupby("loan_status")["last_pymnt_d"]
              .apply(lambda s: round(s.notna().mean(), 4)))
        print("\nnext_pymnt_d non-null rate by loan_status:")
        print(df.groupby("loan_status")["next_pymnt_d"]
              .apply(lambda s: round(s.notna().mean(), 4)))

    section("5. RECOVERY / SETTLEMENT DATA")
    for col in ["recoveries", "collection_recovery_fee", "total_rec_prncp", "out_prncp"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  {col:25s} non-null: {nn:>10,}")

    settlement_cols = [c for c in df.columns if "settlement" in c.lower()]
    if settlement_cols:
        print("\nSettlement-related columns:")
        for col in settlement_cols:
            nn = df[col].notna().sum()
            print(f"  {col:35s} non-null: {nn:>10,}  ({nn/len(df)*100:5.2f}%)")
        if "debt_settlement_flag" in df.columns:
            print("\ndebt_settlement_flag by loan_status:")
            print(pd.crosstab(df["debt_settlement_flag"], df["loan_status"], margins=True))

    section("6. HARDSHIP DATA")
    hardship_cols = [c for c in df.columns if "hardship" in c.lower()]
    if hardship_cols:
        for col in hardship_cols:
            nn = df[col].notna().sum()
            print(f"  {col:40s} non-null: {nn:>10,}  ({nn/len(df)*100:5.2f}%)")
        if "hardship_flag" in df.columns:
            print("\nhardship_flag by loan_status:")
            print(pd.crosstab(df["hardship_flag"], df["loan_status"], margins=True))

    section("7. LEAKAGE COLUMN DETECTION")
    leakage_candidates = [
        "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "next_pymnt_d", "out_prncp",
    ]
    present = [c for c in leakage_candidates if c in df.columns]
    print(f"Leakage columns present: {len(present)}")
    for c in present:
        print(f"  - {c}")

    section("8. OBSERVATION WINDOW FEASIBILITY")
    snapshot = df["issue_d_parsed"].max()
    print(f"Latest issue_d (snapshot proxy): {snapshot.date()}")
    df["months_seasoned"] = ((snapshot - df["issue_d_parsed"]).dt.days / 30.44).round(0)
    print("\nMonths seasoned distribution:")
    print(df["months_seasoned"].describe().round(1))

    print("\nLoans seasoned >= 24 months, by loan_status:")
    seasoned = df[df["months_seasoned"] >= 24]
    print(seasoned["loan_status"].value_counts())
    print(f"\nTotal seasoned >= 24m: {len(seasoned):,}")

    section("DONE")
    print("Review the output above before proceeding to data_prep.py.")


if __name__ == "__main__":
    main()