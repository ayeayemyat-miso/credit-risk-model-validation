"""
lgd.py
------
Phase 3 — Loss Given Default (LGD) estimation.

Methodology
-----------
1. Realised LGD on charged-off loans:
     EAD_at_default = amortisation balance at (months_to_last_pymnt + 6)
     net_recovery   = recoveries - collection_recovery_fee
     LGD_realised   = 1 - net_recovery / EAD_at_default

    This is a scheduled-balance proxy, not a balance observed at charge-off.

2. Logistic-transformed OLS regression:
     logit(LGD) = beta * X + epsilon
     lgd_predicted = 1 / (1 + exp(-X @ beta))

3. Scenario benchmark:
    45% LGD scenario value; applicability to this portfolio is unverified.

Output: data/processed/lgd_scores.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "loans_clean.parquet"
PD_IN = ROOT / "data" / "processed" / "pd_scores.parquet"
OUT = ROOT / "data" / "processed" / "lgd_scores.parquet"

RANDOM_SEED = 42
DEFAULT_LAG_MONTHS = 6
SCENARIO_LGD = 0.45  # Applicability to this portfolio has not been established.


# ---- Feature lists for the LGD regression ----
LGD_NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high",
    "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "tot_hi_cred_lim", "total_bal_ex_mort",
    "avg_cur_bal", "bc_util", "num_actv_rev_tl",
    "num_rev_accts", "pct_tl_nvr_dlq",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mths_since_recent_bc", "mths_since_recent_inq",
    "total_rev_hi_lim", "all_util", "il_util",
]

LGD_CATEGORICAL = [
    "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state", "application_type",
]


def amortisation_balance(funded_amnt, int_rate, term_months, t):
    """
    Outstanding principal after `t` scheduled monthly payments.

    B_t = P (1+r)^t  -  M * ((1+r)^t - 1) / r
    where M is the level monthly payment.
    """
    values = np.asarray([funded_amnt, int_rate, term_months, t], dtype=float)
    if not np.isfinite(values).all() or funded_amnt < 0 or term_months <= 0:
        return np.nan
    if t <= 0:
        return float(funded_amnt)
    if t >= term_months:
        return 0.0
    monthly_rate = int_rate / 12.0
    if monthly_rate == 0:
        return max(float(funded_amnt) * (1 - t / term_months), 0.0)
    if monthly_rate <= -1:
        return np.nan
    log_growth = np.log1p(monthly_rate)
    growth_term = np.expm1(term_months * log_growth)
    payment = (funded_amnt * monthly_rate * np.exp(term_months * log_growth)
               / growth_term)
    balance = (funded_amnt * np.exp(t * log_growth)
               - payment * np.expm1(t * log_growth) / monthly_rate)
    return max(float(balance), 0.0)


def compute_ead_at_default(row):
    """Scheduled-balance proxy at last payment month plus the charge-off lag."""
    months_to_last_payment = pd.to_numeric(
        row["months_to_last_pymnt"], errors="coerce")
    if pd.isna(months_to_last_payment):
        return np.nan
    return amortisation_balance(
        row["funded_amnt"],
        row["int_rate"] / 100.0,
        int(row["term_months"]),
        int(months_to_last_payment + DEFAULT_LAG_MONTHS),
    )


def compute_realised_lgd(row):
    """Compute realised LGD for a single charged-off loan."""
    ead = compute_ead_at_default(row)
    recoveries = pd.to_numeric(row["recoveries"], errors="coerce")
    recovery_fee = pd.to_numeric(row["collection_recovery_fee"], errors="coerce")
    if (not np.isfinite(ead) or ead <= 0 or
            not np.isfinite(recoveries) or not np.isfinite(recovery_fee)):
        return np.nan
    return 1 - (recoveries - recovery_fee) / ead


def make_one_hot_encoder():
    """Support sklearn releases before sparse_output was introduced."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first",
                             sparse_output=True, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first",
                             sparse=True, dtype=np.float64)


def report_metrics(label, observed, predicted):
    rmse = mean_squared_error(observed, predicted) ** 0.5
    bias = float(np.mean(predicted - observed))
    print(f"  {label}: n={len(observed):,}; mean observed={np.mean(observed):.4f}; "
          f"mean predicted={np.mean(predicted):.4f}; bias={bias:+.4f}; "
          f"MAE={mean_absolute_error(observed, predicted):.4f}; "
          f"RMSE={rmse:.4f}; R²={r2_score(observed, predicted):.4f}")


def report_lgd_distribution(label, values):
    values = pd.Series(values, dtype=float)
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"  {label}: n={len(values):,}; mean={values.mean():.4f}; "
          f"exact 0={int(values.eq(0).sum()):,} ({values.eq(0).mean():.2%}); "
          f"exact 1={int(values.eq(1).sum()):,} ({values.eq(1).mean():.2%}); "
          f">0.95={int(values.gt(0.95).sum()):,} ({values.gt(0.95).mean():.2%})")
    print("    Quantiles: " + ", ".join(
        f"p{int(q * 100):02d}={value:.4f}" for q, value in quantiles.items()))


def report_target_cohort_drift(co):
    valid = co.loc[co["lgd_realised"].notna()]
    aggregations = {
        "loans": ("id", "size"),
        "mean_lgd": ("lgd_realised", "mean"),
        "exact_one_share": ("lgd_realised", lambda values: values.eq(1).mean()),
        "above_95_share": ("lgd_realised", lambda values: values.gt(0.95).mean()),
    }
    print("\n  Valid-target LGD by split and term:")
    print(valid.groupby(["split", "term_months"]).agg(**aggregations).to_string(
        float_format=lambda value: f"{value:.3f}"))
    if "issue_year" in valid.columns:
        print("\n  Valid-target LGD by split and issue year:")
        print(valid.groupby(["split", "issue_year"], dropna=False)
              .agg(**aggregations).to_string(
                  float_format=lambda value: f"{value:.3f}"))


def report_prediction_distribution(label, values):
    values = pd.Series(values, dtype=float)
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"  {label}: n={len(values):,}; mean={values.mean():.4f}; "
          f">0.95={int(values.gt(0.95).sum()):,} ({values.gt(0.95).mean():.2%})")
    print("    Quantiles: " + ", ".join(
        f"p{int(q * 100):02d}={value:.4f}" for q, value in quantiles.items()))


def report_target_edge_cases(co):
    ead_excluded = co.loc[co["ead_at_default_proxy"].le(0)].copy()
    ead_excluded["proxy_month_at_or_after_term"] = (
        ead_excluded["default_month_proxy"] >= ead_excluded["term_months"])
    print("\n  EAD <= 0 exclusions by term and proxy default month:")
    ead_table = (ead_excluded.groupby(["term_months", "default_month_proxy"])
                 .agg(loans=("id", "size"),
                      net_recovery_median=("net_recovery", "median"),
                      net_recovery_max=("net_recovery", "max")))
    print(ead_table.to_string(float_format=lambda value: f"{value:.2f}"))
    print(f"    Proxy month >= contractual term: "
          f"{int(ead_excluded['proxy_month_at_or_after_term'].sum()):,}/"
          f"{len(ead_excluded):,}; net recoveries are positive for "
          f"{int(ead_excluded['net_recovery'].gt(0).sum()):,}. "
          "Recovery/EAD ratio is undefined when proxy EAD is zero.")

    negative_lgd = co.loc[co["lgd_realised_raw"].lt(0)].copy()
    negative_table = (negative_lgd.groupby(["term_months", "default_month_proxy"])
                      .agg(loans=("id", "size"),
                           recovery_ead_median=("recovery_ead_ratio", "median"),
                           recovery_ead_max=("recovery_ead_ratio", "max")))
    print("\n  Negative raw LGD by term and proxy default month:")
    print(negative_table.to_string(float_format=lambda value: f"{value:.3f}"))
    if not negative_lgd.empty:
        print("    Net recovery / proxy EAD quantiles: " +
              negative_lgd["recovery_ead_ratio"].quantile(
                  [0.10, 0.50, 0.90, 0.99]).to_string(
                      float_format=lambda value: f"{value:.3f}").replace("\n", "; "))


def main() -> None:
    print("Loading data ...")
    loans = pd.read_parquet(IN)
    pd_scores = pd.read_parquet(PD_IN)
    print(f"  Source loans: {len(loans):,}")
    print(f"  PD set: {len(pd_scores):,}")

    for frame, label in ((loans, "source"), (pd_scores, "PD")):
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{label} loan IDs must be non-null and unique.")
    if not pd_scores["split"].isin(["train", "test"]).all():
        raise ValueError("PD split must contain only 'train' or 'test'.")

    # The PD file defines both the population and the train/test assignment.
    source = loans.copy()
    if "split" in source.columns:
        source = source.rename(columns={"split": "split_source"})
    portfolio = pd_scores[["id", "split"]].merge(
        source, on="id", how="left", validate="one_to_one", indicator=True)
    unmatched = portfolio["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum()):,} PD loan IDs are missing from source data.")
    if "split_source" in portfolio:
        split_mismatch = portfolio["split"].ne(portfolio["split_source"])
        if split_mismatch.any():
            raise ValueError(f"{int(split_mismatch.sum()):,} source and PD splits disagree.")
        portfolio = portfolio.drop(columns="split_source")
    portfolio = portfolio.drop(columns="_merge")
    if portfolio["id"].isna().any() or portfolio["id"].duplicated().any():
        raise AssertionError("Joined loan IDs must remain non-null and unique.")
    print(f"  Authoritative PD universe joined one-to-one: {len(portfolio):,} loans")

    # ---- Step 1: realised LGD on charged-off loans ----
    print("\nComputing realised LGD on charged-off loans ...")
    co = portfolio.loc[portfolio["loan_status"].eq("Charged Off")].copy()
    print(f"  Charged-off loans in PD universe: {len(co):,}")

    required_target_cols = [
        "funded_amnt", "int_rate", "term_months", "months_to_last_pymnt",
        "recoveries", "collection_recovery_fee",
    ]
    missing_target_cols = [c for c in required_target_cols if c not in co.columns]
    if missing_target_cols:
        raise ValueError(f"Missing realised-LGD inputs: {missing_target_cols}")

    co["ead_at_default_proxy"] = co.apply(compute_ead_at_default, axis=1)
    co["default_month_proxy"] = np.trunc(
        pd.to_numeric(co["months_to_last_pymnt"], errors="coerce")
        + DEFAULT_LAG_MONTHS)
    recoveries = pd.to_numeric(co["recoveries"], errors="coerce")
    recovery_fee = pd.to_numeric(co["collection_recovery_fee"], errors="coerce")
    co["net_recovery"] = recoveries - recovery_fee
    valid_ead = np.isfinite(co["ead_at_default_proxy"])
    valid_recoveries = np.isfinite(recoveries)
    valid_fees = np.isfinite(recovery_fee)
    positive_ead = valid_ead & co["ead_at_default_proxy"].gt(0)
    target_inputs_valid = positive_ead & valid_recoveries & valid_fees

    print("  Recovery/EAD input audit:")
    print(f"    Missing recoveries: {int(recoveries.isna().sum()):,}; "
          f"negative recoveries: {int(recoveries.lt(0).sum()):,}; "
          f"zero recoveries: {int(recoveries.eq(0).sum()):,}")
    print(f"    Missing collection fees: {int(recovery_fee.isna().sum()):,}; "
          f"negative fees: {int(recovery_fee.lt(0).sum()):,}; "
          f"zero fees: {int(recovery_fee.eq(0).sum()):,}")
    print(f"    Missing/non-finite EAD proxy: {int((~valid_ead).sum()):,}; "
          f"EAD <= 0: {int((valid_ead & ~co['ead_at_default_proxy'].gt(0)).sum()):,}")
    print(f"    Excluded from realised target: {int((~target_inputs_valid).sum()):,}")

    co["lgd_realised_raw"] = np.nan
    co["recovery_ead_ratio"] = np.nan
    co.loc[target_inputs_valid, "recovery_ead_ratio"] = (
        co.loc[target_inputs_valid, "net_recovery"]
        / co.loc[target_inputs_valid, "ead_at_default_proxy"]
    )
    co.loc[target_inputs_valid, "lgd_realised_raw"] = (
        1 - co.loc[target_inputs_valid, "recovery_ead_ratio"]
    )
    raw_finite = np.isfinite(co["lgd_realised_raw"])
    target_valid = target_inputs_valid & raw_finite
    n_lt0 = int((target_valid & co["lgd_realised_raw"].lt(0)).sum())
    n_gt1 = int((target_valid & co["lgd_realised_raw"].gt(1)).sum())
    print(f"    Non-finite realised LGD after calculation: "
          f"{int((target_inputs_valid & ~raw_finite).sum()):,}")
    print(f"    Raw LGD below 0 / above 1 (clipped): {n_lt0:,} / {n_gt1:,}")
    co["lgd_realised"] = co["lgd_realised_raw"].where(target_valid).clip(0, 1)
    report_target_edge_cases(co)

    print("\n  LGD_realised describe (clipped):")
    print(co["lgd_realised"].describe().round(4))
    report_target_cohort_drift(co)

    ead_related = [
        "funded_amnt", "installment", "int_rate", "term_months",
        "months_to_last_pymnt", "last_pymnt_d", "out_prncp", "total_pymnt",
        "total_rec_prncp", "recoveries", "collection_recovery_fee",
    ]
    available_ead_fields = [c for c in ead_related if c in portfolio.columns]
    print("\n  EAD proxy audit:")
    print(f"    Available related fields: {', '.join(available_ead_fields)}")
    print("    No payment-level history or verified balance-at-charge-off field is available.")
    print("    The proxy assumes scheduled amortisation through last payment + 6 months; "
          "missed payments, accrued interest, and fees can make actual exposure higher.")
    print("    out_prncp/total_pymnt/total_rec_prncp are snapshot or cumulative fields, "
          "not a default-date balance; they are not substituted for EAD.")

    # ---- Step 2: LGD regression ----
    print("\nFitting logistic-transformed OLS on training charged-off loans only ...")
    reg_cols = [c for c in LGD_NUMERIC if c in portfolio.columns]
    cat_cols = [c for c in LGD_CATEGORICAL if c in portfolio.columns]
    if not reg_cols or not cat_cols:
        raise ValueError("Numeric and categorical LGD features are required.")
    missing_features = [c for c in LGD_NUMERIC + LGD_CATEGORICAL
                        if c not in portfolio.columns]
    if missing_features:
        raise ValueError(f"Missing configured LGD features: {missing_features}")
    for col in reg_cols:
        portfolio[col] = pd.to_numeric(portfolio[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        co[col] = pd.to_numeric(co[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)

    train_reg = co.loc[co["split"].eq("train") & co["lgd_realised"].notna()].copy()
    test_reg = co.loc[co["split"].eq("test") & co["lgd_realised"].notna()].copy()
    no_train_values = [
        col for col in reg_cols if not train_reg[col].notna().any()
    ]
    if no_train_values:
        print("  Excluding numeric features entirely missing in training targets: "
              f"{no_train_values}")
        reg_cols = [col for col in reg_cols if col not in no_train_values]
    post_outcome_fields = {
        "loan_status", "default", "months_to_last_pymnt", "recoveries",
        "collection_recovery_fee", "out_prncp", "total_pymnt",
        "total_rec_prncp", "settlement_date", "settlement_amount",
        "settlement_percentage",
    }
    feature_leakage = sorted(post_outcome_fields.intersection(reg_cols + cat_cols))
    print(f"  Post-outcome fields used as predictors: {feature_leakage or 'none'}")
    print("  Predictor fields are application/origination attributes by design; "
          "their as-of timestamps are not present to independently verify timing.")
    print(f"  Train charged-off targets: {len(train_reg):,} "
          f"(of {int(co['split'].eq('train').sum()):,} charged-off loans)")
    print(f"  Test charged-off targets:  {len(test_reg):,} "
          f"(of {int(co['split'].eq('test').sum()):,} charged-off loans)")
    if train_reg.empty or test_reg.empty:
        raise ValueError("Both train and test charged-off targets are required.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", make_one_hot_encoder()),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, reg_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])

    # Fit the regression in logit space; report metrics on the clipped LGD scale.
    eps = 1e-4
    train_target = train_reg["lgd_realised"].to_numpy(dtype=float)
    train_target_logit = np.log(np.clip(train_target, eps, 1 - eps)) - np.log1p(
        -np.clip(train_target, eps, 1 - eps))
    model.fit(train_reg[reg_cols + cat_cols], train_target_logit)

    train_predicted = expit(model.predict(train_reg[reg_cols + cat_cols]))
    test_target = test_reg["lgd_realised"].to_numpy(dtype=float)
    test_predicted = expit(model.predict(test_reg[reg_cols + cat_cols]))
    print("  Metrics use clipped realised LGD as a 0-1 fraction; MAE/RMSE are "
          "LGD-fraction errors (0.10 = 10 percentage points), R² is dimensionless.")
    print("  Valid-target realised LGD distribution:")
    report_lgd_distribution("Train", train_target)
    report_lgd_distribution("Test", test_target)
    report_metrics("Train logit-OLS challenger (in-sample; not validation)",
                   train_target, train_predicted)
    report_metrics("Test logit-OLS challenger (out-of-sample)",
                   test_target, test_predicted)

    training_mean = float(train_target.mean())
    if not np.isfinite(training_mean) or not 0 <= training_mean <= 1:
        raise AssertionError("Training-mean LGD baseline must be finite and within [0, 1].")
    mean_baseline_test = np.full(len(test_target), training_mean)
    print(f"  Selected primary model: training-mean baseline fitted on "
          f"{len(train_target):,} valid train charge-offs; value={training_mean:.4f}")
    report_metrics("Test selected training-mean baseline",
                   test_target, mean_baseline_test)

    # Diagnostic only: isolate the effect of modeling the clipped response directly.
    direct_scale_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regression", LinearRegression()),
    ])
    direct_scale_model.fit(train_reg[reg_cols + cat_cols], train_target)
    direct_scale_test = np.clip(
        direct_scale_model.predict(test_reg[reg_cols + cat_cols]), 0, 1)
    report_metrics("Test direct-scale OLS challenger (clipped; not selected)",
                   test_target, direct_scale_test)
    report_prediction_distribution("Test logit-OLS predictions on valid charge-offs",
                                   test_predicted)
    report_prediction_distribution("Test direct-scale OLS sensitivity",
                                   direct_scale_test)
    fitted_encoder = (model.named_steps["preprocessor"].named_transformers_
                      ["categorical"].named_steps["encoder"])
    print("  Preprocessing fitted only on train targets; fitted category counts and "
          "unseen test loan counts:")
    for col, categories in zip(cat_cols, fitted_encoder.categories_):
        test_values = test_reg[col].fillna("Unknown")
        unknown_count = int((~test_values.isin(categories)).sum())
        print(f"    {col}: {len(categories):,} train categories; "
              f"{unknown_count:,} unseen test loans (ignored by encoder)")

    # ---- Step 3: Score all loans ----
    print("\nScoring all loans in the authoritative PD universe ...")
    df_full = portfolio.copy()
    df_full["lgd_predicted"] = training_mean
    report_prediction_distribution("All-loan selected training-mean baseline predictions",
                                   df_full["lgd_predicted"])

    # Attach realised LGD where available
    df_full = df_full.merge(
        co[["id", "lgd_realised"]],
        on="id", how="left", validate="one_to_one",
    )

    df_full["lgd_regulatory"] = SCENARIO_LGD
    if df_full["id"].isna().any() or df_full["id"].duplicated().any():
        raise AssertionError("LGD output loan IDs must be non-null and unique.")
    output_identity = pd_scores[["id", "split"]].merge(
        df_full[["id", "split"]], on="id", how="outer",
        validate="one_to_one", suffixes=("_pd", "_lgd"), indicator=True)
    if (not output_identity["_merge"].eq("both").all()
            or not output_identity["split_pd"].eq(output_identity["split_lgd"]).all()):
        raise AssertionError("LGD output must preserve every PD ID and train/test split.")
    predicted = df_full["lgd_predicted"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise AssertionError("Predicted LGDs must be finite.")
    if not ((predicted >= 0) & (predicted <= 1)).all():
        raise AssertionError("Predicted LGDs must be within [0, 1].")
    if not np.allclose(predicted, training_mean, rtol=0, atol=1e-12):
        raise AssertionError("Primary predictions must equal the selected training mean.")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("LGD SUMMARY")
    print("=" * 70)
    print("  Selected primary model: train-only clipped realized-LGD mean")
    print(f"  45% scenario benchmark (applicability unverified): {SCENARIO_LGD:.2%}")
    print(f"  Mean predicted LGD (all loans):    {df_full['lgd_predicted'].mean():.4%}")
    print(f"  Mean realised LGD (charged off):   {co['lgd_realised'].mean():.4%}")
    print("  The realized target proxy and combined ECL model are not fully validated.")

    print("\n  LGD by sub_grade (charged-off loans, clipped realised targets):")
    by_grade = (
        co.loc[co["lgd_realised"].notna()].groupby("sub_grade")
        .agg(n=("id", "size"), lgd_mean=("lgd_realised", "mean"))
        .round(4)
        .sort_index()
    )
    print(by_grade.to_string())

    # ---- Save ----
    out_cols = [
        "id", "sub_grade", "default", "split", "loan_status",
        "lgd_realised", "lgd_predicted", "lgd_regulatory",
    ]
    out = df_full[out_cols].copy()

    print(f"\nSaving to {OUT} ...")
    out.to_parquet(OUT, index=False)
    print(f"  Saved. Shape: {out.shape}")

    print("\n  Sample output:")
    print(out.head(10).to_string())


if __name__ == "__main__":
    main()