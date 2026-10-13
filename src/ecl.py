"""Illustrative origination-based ECL using existing PD, LGD, and EAD outputs.

This is not a reporting-date or IFRS 9-compliant accounting allowance. PDs and
EAD are origination-based proxies; LGD is the selected portfolio-wide
training-mean baseline. Contractual loan interest rate is used only as an
explicit discount proxy, not as a verified IFRS 9 effective interest rate.

Run from the project root: python src/ecl.py
Inputs: pd_monthly_hazards, pd_scores, lgd_scores, ead_schedules, staging,
        and loans_clean parquet files under data/processed.
Output: data/processed/ecl_results.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PD_HAZARDS_IN = PROCESSED / "pd_monthly_hazards.parquet"
PD_SCORES_IN = PROCESSED / "pd_scores.parquet"
LGD_IN = PROCESSED / "lgd_scores.parquet"
EAD_IN = PROCESSED / "ead_schedules.parquet"
STAGING_IN = PROCESSED / "staging.parquet"
LOANS_IN = PROCESSED / "loans_clean.parquet"
OUT = PROCESSED / "ecl_results.parquet"

EXPECTED_LOAN_COUNT = 66_688
PD_RECONCILIATION_ATOL = 1e-6
ECL_ORDERING_ATOL = 1e-8
VALID_TERMS = (36, 60)
RATE_PROXY_BASIS = (
    "Contractual int_rate nominal annual percentage used as discount proxy; "
    "not verified IFRS 9 effective interest rate."
)
CALCULATION_BASIS = (
    "Illustrative origination-based ECL; not reporting-date IFRS 9 ECL. "
    "Monthly conditional hazards, contractual start-of-month EAD, selected "
    "training-mean LGD baseline, and contractual-rate discount proxy."
)


def require_unique_ids(frame, label):
    if "id" not in frame.columns:
        raise ValueError(f"{label} is missing required column 'id'.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{label} IDs must be non-null and unique.")


def require_exact_ids(frame, expected_ids, label):
    require_unique_ids(frame, label)
    actual = pd.Index(frame["id"])
    if len(actual) != len(expected_ids):
        raise ValueError(
            f"{label} has {len(actual):,} loans; expected {len(expected_ids):,}."
        )
    missing = expected_ids.difference(actual)
    extra = actual.difference(expected_ids)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} ID mismatch: {len(missing):,} missing, {len(extra):,} extra."
        )


def load_universe():
    """Load loan-level inputs and preserve the PD score universe as authority."""
    pd_scores = pd.read_parquet(PD_SCORES_IN)
    if len(pd_scores) != EXPECTED_LOAN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOAN_COUNT:,} PD scores; found {len(pd_scores):,}."
        )
    require_unique_ids(pd_scores, "PD scores")

    pd_columns = [
        "id", "split", "term_months", "loan_status", "pd_12m", "pd_lifetime",
    ]
    missing_pd = [column for column in pd_columns if column not in pd_scores]
    if missing_pd:
        raise ValueError(f"PD scores are missing required columns: {missing_pd}")
    if not pd_scores["split"].isin(["train", "test"]).all():
        raise ValueError("PD split must contain only 'train' or 'test'.")
    if not pd_scores["term_months"].isin(VALID_TERMS).all():
        raise ValueError(f"PD scores contain terms outside {VALID_TERMS}.")

    loans = pd.read_parquet(
        LOANS_IN,
        columns=["id", "split", "term_months", "loan_status", "int_rate"],
    )
    require_unique_ids(loans, "Clean loans")
    source = loans.rename(columns={
        "split": "split_source",
        "term_months": "term_months_source",
        "loan_status": "loan_status_source",
    })
    universe = pd_scores[pd_columns].merge(
        source,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    source_missing = universe["_merge"].ne("both")
    if source_missing.any():
        raise ValueError(
            f"{int(source_missing.sum()):,} PD IDs are missing from clean loans."
        )
    for pd_col, source_col, label in (
        ("split", "split_source", "split"),
        ("term_months", "term_months_source", "term"),
        ("loan_status", "loan_status_source", "loan status"),
    ):
        mismatches = universe[pd_col].ne(universe[source_col])
        if mismatches.any():
            raise ValueError(
                f"PD/source {label} mismatch for {int(mismatches.sum()):,} loans."
            )
    universe = universe.drop(columns=[
        "split_source", "term_months_source", "loan_status_source", "_merge",
    ])

    lgd = pd.read_parquet(LGD_IN)
    require_exact_ids(lgd, pd.Index(pd_scores["id"]), "LGD scores")
    lgd_columns = ["id", "split", "loan_status", "lgd_predicted"]
    missing_lgd = [column for column in lgd_columns if column not in lgd]
    if missing_lgd:
        raise ValueError(f"LGD scores are missing required columns: {missing_lgd}")
    universe = universe.merge(
        lgd[lgd_columns].rename(columns={
            "split": "split_lgd",
            "loan_status": "loan_status_lgd",
        }),
        on="id",
        how="left",
        validate="one_to_one",
    )
    if (universe["split"].ne(universe["split_lgd"]).any()
            or universe["loan_status"].ne(universe["loan_status_lgd"]).any()):
        raise ValueError("LGD output split/status do not match the PD universe.")
    universe = universe.drop(columns=["split_lgd", "loan_status_lgd"])

    staging = pd.read_parquet(STAGING_IN)
    require_exact_ids(staging, pd.Index(pd_scores["id"]), "Staging output")
    staging_columns = [
        "id", "split", "term_months", "loan_status",
        "observed_classification", "synthetic_stage",
    ]
    missing_staging = [column for column in staging_columns if column not in staging]
    if missing_staging:
        raise ValueError(f"Staging output is missing required columns: {missing_staging}")
    universe = universe.merge(
        staging[staging_columns].rename(columns={
            "split": "split_staging",
            "term_months": "term_months_staging",
            "loan_status": "loan_status_staging",
        }),
        on="id",
        how="left",
        validate="one_to_one",
    )
    for pd_col, stage_col, label in (
        ("split", "split_staging", "split"),
        ("term_months", "term_months_staging", "term"),
        ("loan_status", "loan_status_staging", "loan status"),
    ):
        if universe[pd_col].ne(universe[stage_col]).any():
            raise ValueError(f"Staging output {label} differs from PD scores.")

    expected_observed_classification = universe["loan_status"].map({
        "Fully Paid": "Closed",
        "Charged Off": "Historical Stage 3 proxy",
        "Current": "Unassessed",
    })
    if (expected_observed_classification.isna().any()
            or not expected_observed_classification.eq(
                universe["observed_classification"]
            ).all()):
        raise ValueError("Observed staging labels do not match loan_status.")
    noncurrent_synthetic = (
        universe["loan_status"].ne("Current") & universe["synthetic_stage"].notna()
    )
    if noncurrent_synthetic.any():
        raise ValueError("Synthetic stages must be null for non-Current loans.")
    unexpected_stages = set(universe["synthetic_stage"].dropna()).difference({
        "Synthetic Stage 1", "Synthetic Stage 2",
    })
    if unexpected_stages:
        raise ValueError(f"Unexpected synthetic stage labels: {sorted(unexpected_stages)}")

    universe = universe.rename(columns={
        "lgd_predicted": "lgd_baseline",
        "observed_classification": "observed_classification",
    })
    universe = universe.drop(columns=[
        "split_staging", "term_months_staging", "loan_status_staging",
    ])

    rates = pd.to_numeric(universe["int_rate"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid_rates = ~np.isfinite(rates) | (rates < 0) | (rates > 100)
    print("Contractual-rate discount proxy audit:")
    print(f"  Annual int_rate percent: min={np.nanmin(rates):.4f}; "
          f"median={np.nanmedian(rates):.4f}; max={np.nanmax(rates):.4f}")
    print(f"  Missing/non-finite/negative/>100% rates: {int(invalid_rates.sum()):,}")
    if invalid_rates.any():
        raise ValueError(
            "Aborting: ECL discount proxy requires finite annual nominal "
            "int_rate percentages in [0, 100]."
        )
    universe["contractual_int_rate_pct"] = rates
    universe["lgd_baseline"] = pd.to_numeric(
        universe["lgd_baseline"], errors="coerce"
    )
    lgd_values = universe["lgd_baseline"].to_numpy(dtype=np.float64)
    if not np.isfinite(lgd_values).all() or (lgd_values < 0).any() or (lgd_values > 1).any():
        raise ValueError("LGD baseline must be finite and within [0, 1].")

    expected_ids = pd.Index(pd_scores["id"])
    if len(universe) != EXPECTED_LOAN_COUNT:
        raise AssertionError("Loan-level input joins changed the authoritative PD row count.")
    require_exact_ids(universe, expected_ids, "Joined ECL universe")
    return universe, pd_scores


def validate_monthly_frame(frame, expected_ids, term, label):
    if frame["id"].isna().any() or frame.duplicated(["id", "month"]).any():
        raise ValueError(f"{label} contains missing IDs or duplicate (id, month) keys.")
    if not frame["term_months"].eq(term).all():
        raise ValueError(f"{label} contains a term inconsistent with filter {term}.")
    if not frame["month"].between(1, term).all():
        raise ValueError(f"{label} contains months outside 1..{term}.")
    term_ids = pd.Index(frame["id"].unique())
    if len(term_ids.difference(expected_ids)) or len(expected_ids.difference(term_ids)):
        raise ValueError(f"{label} IDs do not exactly match PD loans with term {term}.")
    counts = frame.groupby("id", sort=False).size()
    if not counts.eq(term).all():
        raise ValueError(f"{label} does not contain exactly {term} rows per loan.")


def calculate_term_ecl(universe, term, tolerance):
    """Calculate and aggregate one term's monthly probabilities and losses."""
    hazards = pd.read_parquet(
        PROCESSED / "pd_monthly_hazards.parquet",
        columns=["id", "split", "term_months", "month", "hazard"],
        filters=[("term_months", "==", term)],
    )
    ead = pd.read_parquet(
        EAD_IN,
        columns=["id", "split", "term_months", "month", "ead"],
        filters=[("term_months", "==", term)],
    )
    expected_term = universe.loc[universe["term_months"].eq(term)]
    expected_ids = pd.Index(expected_term["id"])
    validate_monthly_frame(hazards, expected_ids, term, "Monthly hazards")
    validate_monthly_frame(ead, expected_ids, term, "EAD schedules")

    if len(hazards) != len(ead):
        raise ValueError(f"Term {term}: hazard/EAD row counts differ.")
    if hazards.duplicated(["id", "month"]).any() or ead.duplicated(["id", "month"]).any():
        raise ValueError(f"Term {term}: duplicate monthly join keys.")

    monthly = hazards.merge(
        ead.rename(columns={
            "split": "split_ead",
            "term_months": "term_months_ead",
        }),
        on=["id", "month"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = monthly["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(
            f"Term {term}: {int(unmatched.sum()):,} hazard keys lack matching EAD."
        )
    if (monthly["split"].ne(monthly["split_ead"]).any()
            or monthly["term_months"].ne(monthly["term_months_ead"]).any()):
        raise ValueError(f"Term {term}: hazard and EAD split/term values disagree.")
    monthly = monthly.drop(columns=["_merge", "split_ead", "term_months_ead"])

    monthly = monthly.sort_values(["id", "month"], kind="stable").reset_index(drop=True)
    if monthly.duplicated(["id", "month"]).any():
        raise AssertionError(f"Term {term}: duplicate loan-month keys after joining.")

    hazard = pd.to_numeric(monthly["hazard"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    exposure = pd.to_numeric(monthly["ead"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(hazard).all() or (hazard < 0).any() or (hazard > 1).any():
        raise ValueError(f"Term {term}: hazards must be finite and within [0, 1].")
    if not np.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError(f"Term {term}: EAD must be finite and nonnegative.")

    loan_lgd = expected_term.set_index("id")["lgd_baseline"]
    loan_rate = expected_term.set_index("id")["contractual_int_rate_pct"]
    lgd = monthly["id"].map(loan_lgd).to_numpy(dtype=np.float64)
    annual_rate_pct = monthly["id"].map(loan_rate).to_numpy(dtype=np.float64)
    if (not np.isfinite(lgd).all() or (lgd < 0).any() or (lgd > 1).any()
            or not np.isfinite(annual_rate_pct).all()):
        raise ValueError(f"Term {term}: invalid mapped LGD or contractual rate.")

    group_ids = monthly["id"]
    survival_after = pd.Series(1.0 - hazard).groupby(
        group_ids, sort=False
    ).cumprod()
    survival_before = survival_after.groupby(group_ids, sort=False).shift(
        1, fill_value=1.0
    )
    marginal_pd = hazard * survival_before.to_numpy(dtype=np.float64)
    discount_factor = np.power(
        1.0 + annual_rate_pct / 1200.0,
        -monthly["month"].to_numpy(dtype=np.float64),
    )
    if not np.isfinite(survival_before.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Term {term}: survival values are non-finite.")
    if (survival_before.lt(-tolerance).any()
            or survival_before.gt(1.0 + tolerance).any()):
        raise ValueError(f"Term {term}: survival values fall outside [0, 1].")
    if (not np.isfinite(marginal_pd).all()
            or (marginal_pd < -tolerance).any()
            or (marginal_pd > 1.0 + tolerance).any()):
        raise ValueError(f"Term {term}: marginal default probabilities are invalid.")
    if not np.isfinite(discount_factor).all() or (discount_factor <= 0).any() \
            or (discount_factor > 1.0 + tolerance).any():
        raise ValueError(f"Term {term}: discount factors must be finite in (0, 1].")

    monthly_loss = marginal_pd * lgd * exposure * discount_factor
    if not np.isfinite(monthly_loss).all() or (monthly_loss < 0).any():
        raise ValueError(f"Term {term}: monthly ECL contributions are invalid.")

    monthly["marginal_pd"] = marginal_pd
    monthly["monthly_loss"] = monthly_loss
    lifetime_aggregation = monthly.groupby("id", sort=False).agg(
        pd_lifetime_reconstructed=("marginal_pd", "sum"),
        ecl_lifetime=("monthly_loss", "sum"),
    )
    pd12_aggregation = (monthly.loc[monthly["month"].le(12)]
                        .groupby("id", sort=False)["marginal_pd"].sum()
                        .rename("pd_12m_reconstructed"))
    ecl12_aggregation = (monthly.loc[monthly["month"].le(12)]
                         .groupby("id", sort=False)["monthly_loss"].sum()
                         .rename("ecl_12m"))
    aggregation = lifetime_aggregation.join(pd12_aggregation, validate="one_to_one")
    aggregation = aggregation.join(ecl12_aggregation, validate="one_to_one")
    aggregation = aggregation.reset_index()

    reference = expected_term.set_index("id").loc[aggregation["id"]]
    lifetime_error = np.abs(
        aggregation["pd_lifetime_reconstructed"].to_numpy(dtype=np.float64)
        - reference["pd_lifetime"].to_numpy(dtype=np.float64)
    )
    pd12_error = np.abs(
        aggregation["pd_12m_reconstructed"].to_numpy(dtype=np.float64)
        - reference["pd_12m"].to_numpy(dtype=np.float64)
    )
    if (lifetime_error > PD_RECONCILIATION_ATOL).any():
        raise AssertionError(
            f"Term {term}: lifetime PD does not reconcile; "
            f"max absolute error={lifetime_error.max():.3e}."
        )
    if (pd12_error > PD_RECONCILIATION_ATOL).any():
        raise AssertionError(
            f"Term {term}: 12-month PD does not reconcile; "
            f"max absolute error={pd12_error.max():.3e}."
        )
    if not np.isfinite(aggregation[["ecl_12m", "ecl_lifetime"]].to_numpy()).all():
        raise AssertionError(f"Term {term}: aggregated ECL must be finite.")
    if (aggregation[["ecl_12m", "ecl_lifetime"]] < 0).any().any():
        raise AssertionError(f"Term {term}: aggregated ECL must be nonnegative.")
    if (aggregation["ecl_12m"] > aggregation["ecl_lifetime"] + ECL_ORDERING_ATOL).any():
        raise AssertionError(f"Term {term}: 12-month ECL exceeds lifetime ECL.")

    diagnostics = {
        "loans": len(aggregation),
        "monthly_rows": len(monthly),
        "lifetime_max_abs_error": float(lifetime_error.max()),
        "pd12_max_abs_error": float(pd12_error.max()),
        "hazard_min": float(hazard.min()),
        "hazard_max": float(hazard.max()),
        "marginal_min": float(marginal_pd.min()),
        "marginal_max": float(marginal_pd.max()),
        "ead_min": float(exposure.min()),
        "ead_max": float(exposure.max()),
        "lgd_min": float(lgd.min()),
        "lgd_max": float(lgd.max()),
        "df_min": float(discount_factor.min()),
        "df_max": float(discount_factor.max()),
    }
    del monthly, survival_after, survival_before, marginal_pd, monthly_loss
    return aggregation, diagnostics


def main():
    print("Loading authoritative loan universe and loan-level assumptions ...")
    universe, pd_scores = load_universe()
    pd_ids = pd.Index(pd_scores["id"])
    if universe["id"].nunique() != EXPECTED_LOAN_COUNT:
        raise AssertionError("ECL universe must contain exactly 66,688 unique PD IDs.")
    print(f"  Authoritative PD loans: {len(universe):,}; unique IDs: "
          f"{universe['id'].nunique():,}")
    print(f"  Loan-status counts: {universe['loan_status'].value_counts().to_dict()}")
    print(f"  LGD baseline range: [{universe['lgd_baseline'].min():.6f}, "
          f"{universe['lgd_baseline'].max():.6f}]")

    print(f"  Discount convention: {RATE_PROXY_BASIS}")
    print("  Monthly timing: default in month m is paired with start-of-month EAD "
          "and discounted m periods as a month-end proxy; intra-month default/"
          "recovery timing, fees, and verified EIR cash flows are unavailable.")

    term_aggregations = []
    diagnostics_by_term = {}
    for term in VALID_TERMS:
        aggregation, diagnostics = calculate_term_ecl(
            universe, term, tolerance=1e-12
        )
        term_aggregations.append(aggregation)
        diagnostics_by_term[term] = diagnostics
        print(f"  Term {term}: loans={diagnostics['loans']:,}; "
              f"loan-month rows={diagnostics['monthly_rows']:,}; "
              f"max PD errors (12m/lifetime)="
              f"{diagnostics['pd12_max_abs_error']:.3e}/"
              f"{diagnostics['lifetime_max_abs_error']:.3e}")

    aggregate = pd.concat(term_aggregations, ignore_index=True)
    result = universe.merge(
        aggregate,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise AssertionError("Not every PD loan received aggregated ECL measures.")
    result = result.drop(columns="_merge")

    current = result["loan_status"].eq("Current")
    stage1 = current & result["synthetic_stage"].eq("Synthetic Stage 1")
    stage2 = current & result["synthetic_stage"].eq("Synthetic Stage 2")
    valid_current_stage = stage1 | stage2
    selected_ecl = np.full(len(result), np.nan, dtype=np.float64)
    selected_ecl[stage1.to_numpy()] = result.loc[stage1, "ecl_12m"].to_numpy(
        dtype=np.float64
    )
    selected_ecl[stage2.to_numpy()] = result.loc[stage2, "ecl_lifetime"].to_numpy(
        dtype=np.float64
    )
    result["ecl_selected_synthetic"] = selected_ecl
    result["calculation_basis"] = CALCULATION_BASIS
    result["discount_rate_basis"] = RATE_PROXY_BASIS

    if len(result) != EXPECTED_LOAN_COUNT or result["id"].nunique() != EXPECTED_LOAN_COUNT:
        raise AssertionError("ECL output must contain exactly 66,688 unique PD loans.")
    if result["id"].isna().any() or result["id"].duplicated().any():
        raise AssertionError("ECL output IDs must be non-null and unique.")
    if not np.array_equal(result["split"].to_numpy(),
                          pd_scores.set_index("id")["split"].reindex(result["id"]).to_numpy()):
        raise AssertionError("ECL output did not preserve authoritative PD splits.")

    ecl_values = result[["ecl_12m", "ecl_lifetime"]].to_numpy(dtype=np.float64)
    if not np.isfinite(ecl_values).all() or (ecl_values < 0).any():
        raise AssertionError("All full-universe ECL values must be finite and nonnegative.")
    if (result["ecl_12m"] > result["ecl_lifetime"] + ECL_ORDERING_ATOL).any():
        raise AssertionError("12-month ECL must not exceed lifetime ECL.")
    selected_mask = result["ecl_selected_synthetic"].notna()
    if (selected_mask & ~valid_current_stage).any():
        raise AssertionError("Selected synthetic ECL may only be assigned to staged Current loans.")
    if not result.loc[valid_current_stage, "ecl_selected_synthetic"].notna().all():
        raise AssertionError("Current loans with synthetic Stage 1/2 require selected ECL.")
    if result.loc[~valid_current_stage, "ecl_selected_synthetic"].notna().any():
        raise AssertionError("Excluded/unassessable loans must retain null selected ECL.")
    selected_values = result.loc[selected_mask, "ecl_selected_synthetic"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(selected_values).all() or (selected_values < 0).any():
        raise AssertionError("Populated selected synthetic ECL values must be finite/nonnegative.")
    if (not np.allclose(
            result.loc[stage1, "ecl_selected_synthetic"],
            result.loc[stage1, "ecl_12m"], rtol=0, atol=ECL_ORDERING_ATOL
        ) or not np.allclose(
            result.loc[stage2, "ecl_selected_synthetic"],
            result.loc[stage2, "ecl_lifetime"], rtol=0, atol=ECL_ORDERING_ATOL
        )):
        raise AssertionError("Selected ECL does not match the assigned synthetic stage.")

    output_columns = [
        "id", "split", "term_months", "loan_status", "observed_classification",
        "synthetic_stage", "pd_12m_reconstructed", "pd_lifetime_reconstructed",
        "lgd_baseline", "contractual_int_rate_pct", "ecl_12m", "ecl_lifetime",
        "ecl_selected_synthetic", "calculation_basis", "discount_rate_basis",
    ]
    output = result[output_columns].copy()

    print("\nECL summary (full-universe hypothetical origination basis):")
    print(f"  Loans processed: {len(output):,}")
    print(f"  12-month ECL: total={output['ecl_12m'].sum():,.2f}; "
          f"mean/loan={output['ecl_12m'].mean():,.4f}")
    print(f"  Lifetime ECL: total={output['ecl_lifetime'].sum():,.2f}; "
          f"mean/loan={output['ecl_lifetime'].mean():,.4f}")
    print("  Current synthetic stage counts:")
    print(result.loc[current, "synthetic_stage"].value_counts(dropna=False).to_string())
    print(f"  Current loans selected: {int(selected_mask.sum()):,}; "
          f"selected illustrative ECL total={np.nansum(selected_ecl):,.2f}")
    print(f"  Null selected ECL for non-current/unassessable rows: "
          f"{int((~selected_mask).sum()):,}; zero-filled excluded rows: "
          f"{int((~selected_mask & result['ecl_selected_synthetic'].eq(0)).sum()):,}")
    print("  PD reconstruction and loan-month key checks passed; maximum errors by term:")
    for term, diagnostics in diagnostics_by_term.items():
        print(f"    {term}m: pd_12m={diagnostics['pd12_max_abs_error']:.3e}; "
              f"pd_lifetime={diagnostics['lifetime_max_abs_error']:.3e}")
    print("  Validation passed: unique IDs, consistent joins, finite/ranged inputs, "
          "nonnegative ECL, 12m <= lifetime, and selected ECL only for Current synthetic stages.")
    print("  Limitations: origination-based PD/EAD, fixed baseline LGD, synthetic SICR stages, "
          "and contractual-rate discount proxy; not IFRS 9 reporting-date allowances.")

    output.to_parquet(OUT, index=False, compression="zstd")
    print(f"Saved illustrative ECL results to {OUT}")


if __name__ == "__main__":
    main()
