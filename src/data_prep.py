"""
data_prep.py
------------
Phase 1 — Data preparation for the credit risk pipeline.

Loads the LendingClub dataset, filters to the sample of interest,
derives timing and target variables, and saves a clean parquet file.

Output: data/processed/loans_clean.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "loan_data.csv"
OUT = ROOT / "data" / "processed" / "loans_clean.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
SNAPSHOT_DATE = pd.Timestamp("2018-12-31")
VINTAGE_START = 2011
VINTAGE_END = 2014
KEEP_STATUSES = ["Fully Paid", "Charged Off", "Current"]
OOT_SPLIT_YEAR = 2014          # test = 2014, train = 2011-2013
MIN_SEASONING_MONTHS = 24

# ---- Columns to load (trimmed set) ----
USECOLS = [
    # Identifier
    "id",

    # Origination-time features
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
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

    # Timing
    "issue_d", "last_pymnt_d", "next_pymnt_d",

    # Outcome / LGD / EAD
    "loan_status",
    "recoveries", "collection_recovery_fee",
    "total_rec_prncp", "out_prncp", "total_pymnt",

    # Settlement (kept for LGD cross-check)
    "debt_settlement_flag", "settlement_date",
    "settlement_amount", "settlement_percentage",
]


def parse_my(date_series: pd.Series) -> pd.Series:
    """Parse LendingClub date strings in 'Mon-YYYY' format."""
    return pd.to_datetime(date_series, format="%b-%Y", errors="coerce")


def months_between(d1: pd.Series, d2: pd.Series) -> pd.Series:
    """Whole months between two datetime series (d2 - d1)."""
    return ((d2.dt.year - d1.dt.year) * 12 + (d2.dt.month - d1.dt.month)).astype("float")


def main() -> None:
    print("Loading raw data ...")
    df = pd.read_csv(RAW, low_memory=False, usecols=USECOLS)
    print(f"  Raw shape: {df.shape}")

    # ---- Parse dates ----
    df["issue_d"] = parse_my(df["issue_d"])
    df["last_pymnt_d"] = parse_my(df["last_pymnt_d"])
    df["next_pymnt_d"] = parse_my(df["next_pymnt_d"])
    df["earliest_cr_line"] = parse_my(df["earliest_cr_line"])
    df["settlement_date"] = parse_my(df["settlement_date"])

    # ---- Filter to sample ----
    df["issue_year"] = df["issue_d"].dt.year
    df = df[df["issue_year"].between(VINTAGE_START, VINTAGE_END)]
    df = df[df["loan_status"].isin(KEEP_STATUSES)]
    print(f"  After vintage + status filter: {df.shape}")

    # ---- Derive timing variables ----
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)
    df["months_to_last_pymnt"] = months_between(df["issue_d"], df["last_pymnt_d"])
    months_to_snapshot = (
        (SNAPSHOT_DATE.year - df["issue_d"].dt.year) * 12
        + (SNAPSHOT_DATE.month - df["issue_d"].dt.month)
    )
    df["months_seasoned"] = months_to_snapshot.astype("float")

    # ---- Target definition ----
    # default = 1 if Charged Off, else 0 (Fully Paid or Current)
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    # Flag which loans are eligible for the PD sample
    df["in_pd_sample"] = df["months_seasoned"] >= MIN_SEASONING_MONTHS

    # ---- OOT split ----
    df["split"] = np.where(df["issue_year"] < OOT_SPLIT_YEAR, "train", "test")

    # ---- Derived numeric fields used downstream ----
    df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SAMPLE SUMMARY")
    print("=" * 70)
    print(f"Snapshot date:           {SNAPSHOT_DATE.date()}")
    print(f"Total loans:             {len(df):,}")
    print(f"Train (2011-2013):       {(df['split'] == 'train').sum():,}")
    print(f"Test  (2014):            {(df['split'] == 'test').sum():,}")
    print(f"\nLoan status:")
    print(df["loan_status"].value_counts())
    print(f"\nDefault rate:            {df['default'].mean():.4%}")
    print(f"PD-eligible loans:       {df['in_pd_sample'].sum():,} "
          f"({df['in_pd_sample'].mean():.2%})")
    print(f"\nLoans per vintage:")
    print(df["issue_year"].value_counts().sort_index())
    print(f"\nSeasoning (months) describe:")
    print(df["months_seasoned"].describe().round(1))

    # ---- Save ----
    print(f"\nSaving to {OUT} ...")
    df.to_parquet(OUT, index=False)
    print(f"  Saved. Shape: {df.shape}")

    # ---- Data-quality flags ----
    print("\n" + "=" * 70)
    print("DATA QUALITY FLAGS")
    print("=" * 70)
    print(f"Missing int_rate:        {df['int_rate'].isna().sum():,}")
    print(f"Missing dti:             {df['dti'].isna().sum():,}")
    print(f"Missing annual_inc:      {df['annual_inc'].isna().sum():,}")
    print(f"Missing last_pymnt_d:    {df['last_pymnt_d'].isna().sum():,}")
    print(f"Missing months_to_last_pymnt: "
          f"{df['months_to_last_pymnt'].isna().sum():,}")
    print(f"Charged Off with no recoveries: "
          f"{((df['loan_status']=='Charged Off') & (df['recoveries']==0)).sum():,}")


if __name__ == "__main__":
    main()
