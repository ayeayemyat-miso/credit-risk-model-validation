"""Out-of-time PD/LGD validation and lightweight portfolio arithmetic checks.

This script validates the existing outputs; it does not retrain models, write
processed parquet files, or establish IFRS 9/Basel/CRR regulatory validation.

Run from the project root: python src/model_validation.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

if __package__:
    from . import lgd as lgd_model
else:
    import lgd as lgd_model

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

LOANS_IN = PROCESSED / "loans_clean.parquet"
PD_IN = PROCESSED / "pd_scores.parquet"
LGD_IN = PROCESSED / "lgd_scores.parquet"
EAD_IN = PROCESSED / "ead_schedules.parquet"
HAZARDS_IN = PROCESSED / "pd_monthly_hazards.parquet"
STAGING_IN = PROCESSED / "staging.parquet"
ECL_IN = PROCESSED / "ecl_results.parquet"
RWA_IN = PROCESSED / "rwa_results.parquet"

SUMMARY_OUT = REPORTS / "model_validation_summary.csv"
DECILES_OUT = REPORTS / "pd_calibration_deciles.csv"
SUBGRADE_OUT = REPORTS / "pd_calibration_subgrade.csv"
LGD_SEGMENTS_OUT = REPORTS / "lgd_validation_segments.csv"
ROC_FIGURE_OUT = FIGURES / "pd_roc_curve.png"
CALIBRATION_FIGURE_OUT = FIGURES / "pd_calibration_plot.png"

EXPECTED_LOAN_COUNT = 66_688
EXPECTED_SPLIT_COUNTS = {"train": 31_460, "test": 35_228}
EXPECTED_12M_TEST = {"eligible": 30_827, "events": 614, "excluded": 4_401}
VALID_TERMS = (36, 60)
PD_TOLERANCE = 1e-6
EAD_TOLERANCE = 1e-7
ECL_TOLERANCE = 1e-8
MIN_SEGMENT_N = 50


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_unique_id(frame, label):
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{label} IDs must be non-null and unique.")


def assert_same_ids(frame, expected_ids, label):
    require_unique_id(frame, label)
    actual_ids = pd.Index(frame["id"])
    if len(actual_ids) != len(expected_ids):
        raise ValueError(
            f"{label} has {len(actual_ids):,} loans; expected {len(expected_ids):,}."
        )
    missing = expected_ids.difference(actual_ids)
    extra = actual_ids.difference(expected_ids)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} ID mismatch: {len(missing):,} missing, {len(extra):,} extra."
        )


def add_summary(rows, section, metric, value, note=""):
    rows.append({
        "section": section,
        "metric": metric,
        "value": value,
        "notes": note,
    })


def check_metadata_alignment(frame, base, label, columns):
    require_columns(frame, ["id"] + columns, label)
    assert_same_ids(frame, pd.Index(base["id"]), label)
    left = base[["id"] + columns]
    right = frame[["id"] + columns]
    joined = left.merge(
        right,
        on="id",
        how="outer",
        validate="one_to_one",
        suffixes=("_pd", "_output"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{label} does not cover the PD ID universe exactly.")
    for column in columns:
        if not joined[f"{column}_pd"].eq(joined[f"{column}_output"]).all():
            mismatch_count = int(
                joined[f"{column}_pd"].ne(joined[f"{column}_output"]).sum()
            )
            raise ValueError(
                f"{label} {column} differs from PD scores for {mismatch_count:,} loans."
            )


def load_inputs():
    pd_scores = pd.read_parquet(PD_IN)
    require_columns(pd_scores, [
        "id", "split", "term_months", "loan_status", "default", "sub_grade", "issue_year",
        "pd_12m", "pd_lifetime", "is_event", "event_month", "obs_end",
    ], "PD scores")
    require_unique_id(pd_scores, "PD scores")
    if len(pd_scores) != EXPECTED_LOAN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOAN_COUNT:,} PD loans; found {len(pd_scores):,}."
        )
    split_counts = pd_scores["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"PD split counts differ from expected train/test cohorts: {split_counts}"
        )
    if not pd_scores["term_months"].isin(VALID_TERMS).all():
        raise ValueError(f"PD terms must be one of {VALID_TERMS}.")

    loan_columns = [
        "id", "split", "term_months", "loan_status", "default", "issue_year",
        "months_to_last_pymnt", "months_seasoned", "sub_grade", "funded_amnt",
        "int_rate", "recoveries", "collection_recovery_fee", "out_prncp",
        "purpose", "application_type",
    ]
    loans = pd.read_parquet(LOANS_IN, columns=loan_columns)
    require_unique_id(loans, "Clean loans")
    loan_source = loans.rename(columns={
        "split": "split_source",
        "term_months": "term_months_source",
        "loan_status": "loan_status_source",
        "default": "default_source",
        "sub_grade": "sub_grade_source",
        "issue_year": "issue_year_source",
        "months_to_last_pymnt": "months_to_last_pymnt_source",
    })
    base = pd_scores.merge(
        loan_source,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = base["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum()):,} PD IDs are missing from loans_clean.")
    for pd_column, source_column, label in (
        ("split", "split_source", "split"),
        ("term_months", "term_months_source", "term"),
        ("loan_status", "loan_status_source", "loan_status"),
        ("default", "default_source", "default"),
        ("sub_grade", "sub_grade_source", "sub_grade"),
        ("issue_year", "issue_year_source", "issue_year"),
        ("months_to_last_pymnt", "months_to_last_pymnt_source", "months_to_last_pymnt"),
    ):
        mismatch = base[pd_column].ne(base[source_column])
        if mismatch.any():
            raise ValueError(
                f"PD and clean-loan {label} differ for {int(mismatch.sum()):,} loans."
            )
    base = base.drop(columns=[
        "split_source", "term_months_source", "loan_status_source",
        "default_source", "sub_grade_source", "issue_year_source",
        "months_to_last_pymnt_source", "_merge",
    ])

    lgd = pd.read_parquet(LGD_IN)
    check_metadata_alignment(lgd, pd_scores, "LGD scores", ["split", "loan_status"])
    require_columns(lgd, ["lgd_realised", "lgd_predicted"], "LGD scores")
    base = base.merge(
        lgd[["id", "lgd_realised", "lgd_predicted"]],
        on="id", how="left", validate="one_to_one",
    )

    staging = pd.read_parquet(STAGING_IN)
    check_metadata_alignment(
        staging, pd_scores, "Staging output", ["split", "term_months", "loan_status"]
    )
    require_columns(staging, ["observed_classification", "synthetic_stage"], "Staging")
    base = base.merge(
        staging[["id", "observed_classification", "synthetic_stage"]],
        on="id", how="left", validate="one_to_one",
    )

    ecl = pd.read_parquet(ECL_IN)
    check_metadata_alignment(ecl, pd_scores, "ECL output", ["split", "term_months", "loan_status"])
    ecl_columns = [
        "pd_12m_reconstructed", "pd_lifetime_reconstructed", "lgd_baseline",
        "contractual_int_rate_pct", "ecl_12m", "ecl_lifetime",
        "ecl_selected_synthetic",
    ]
    require_columns(ecl, ecl_columns, "ECL output")
    base = base.merge(
        ecl[["id"] + ecl_columns], on="id", how="left", validate="one_to_one"
    )

    rwa = pd.read_parquet(RWA_IN)
    check_metadata_alignment(rwa, pd_scores, "RWA output", ["split", "term_months", "loan_status"])
    require_columns(rwa, [
        "out_prncp_snapshot", "risk_weight_assumption", "rwa_illustrative",
        "included_in_current_portfolio", "collateral_recognized", "ccf_assumption",
    ], "RWA output")
    base = base.merge(
        rwa[["id", "out_prncp_snapshot", "risk_weight_assumption",
             "rwa_illustrative", "included_in_current_portfolio",
             "collateral_recognized", "ccf_assumption"]],
        on="id", how="left", validate="one_to_one",
    )
    return base, pd_scores, loans, lgd, staging, ecl, rwa


def validate_loan_level_values(base, summary_rows):
    for column in ("pd_12m", "pd_lifetime", "lgd_predicted"):
        values = pd.to_numeric(base[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"{column} must be finite and within [0, 1] for every loan.")
    realized_mask = base["lgd_realised"].notna()
    realized_values = base.loc[realized_mask, "lgd_realised"].to_numpy(dtype=np.float64)
    if (not np.isfinite(realized_values).all() or (realized_values < 0).any()
            or (realized_values > 1).any()):
        raise ValueError("Non-null realized LGD values must be finite and within [0, 1].")
    if (realized_mask & ~base["loan_status"].eq("Charged Off")).any():
        raise ValueError("Realized LGD is populated for loans not labelled Charged Off.")

    missing = []
    for column in (
        "pd_12m", "pd_lifetime", "is_event", "obs_end", "event_month",
        "lgd_predicted", "lgd_realised", "out_prncp", "int_rate",
    ):
        count = int(base[column].isna().sum())
        missing.append({
            "field": column,
            "missing_count": count,
            "missing_pct": count / len(base),
            "interpretation": (
                "expected for non-Charged-Off loans and excluded proxy targets"
                if column == "lgd_realised"
                else "event month is not assigned to non-Charged-Off loans"
                if column == "event_month"
                else "unexpected missingness"
                if count
                else "complete in PD universe"
            ),
        })
        if column not in {"lgd_realised", "event_month"} and count:
            raise ValueError(f"Unexpected missing values in required field {column}: {count:,}.")
    return pd.DataFrame(missing)


def verify_pd_observation_method(base, summary_rows):
    status = base["loan_status"]
    charge = status.eq("Charged Off")
    paid = status.eq("Fully Paid")
    current = status.eq("Current")
    if not (charge | paid | current).all():
        raise ValueError("Unexpected loan status in PD universe.")

    last_payment_month = pd.to_numeric(base["months_to_last_pymnt"], errors="coerce")
    seasoned = pd.to_numeric(base["months_seasoned"], errors="coerce")
    term = pd.to_numeric(base["term_months"], errors="coerce")
    proxy_event_month = last_payment_month + 6
    expected_event_month = proxy_event_month.where(charge)
    event_month = pd.to_numeric(base["event_month"], errors="coerce")
    if not np.allclose(
        event_month.to_numpy(dtype=np.float64),
        expected_event_month.to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError("event_month is inconsistent with last payment + six-month proxy.")

    raw_end = np.where(charge, proxy_event_month, np.where(paid, last_payment_month, seasoned))
    expected_obs_end = np.floor(np.minimum(np.minimum(raw_end, seasoned), term))
    obs_end = pd.to_numeric(base["obs_end"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(obs_end).all() or not np.allclose(
        obs_end, expected_obs_end, rtol=0, atol=1e-9
    ):
        raise ValueError("obs_end is inconsistent with pd.py observation/censoring rules.")
    if (obs_end < 1).any() or (obs_end > term.to_numpy(dtype=np.float64)).any():
        raise ValueError("Observed follow-up is outside the valid loan interval.")

    expected_is_event = (
        charge
        & proxy_event_month.notna()
        & proxy_event_month.ge(1)
        & proxy_event_month.le(seasoned)
        & proxy_event_month.le(term)
    ).astype(np.int8)
    if not base["is_event"].astype(np.int8).eq(expected_is_event).all():
        raise ValueError("is_event is inconsistent with pd.py event validity rules.")

    summary_rows.extend([
        {"section": "PD endpoint", "metric": "event_month definition", "value": "months_to_last_pymnt + 6", "notes": "Charge-off timing proxy, not verified default date."},
        {"section": "PD endpoint", "metric": "obs_end reconstruction mismatches", "value": 0, "notes": "Recomputed from status, censoring, snapshot seasoning, and contractual term."},
        {"section": "PD endpoint", "metric": "is_event reconstruction mismatches", "value": 0, "notes": "Recomputed using existing pd.py validity conditions."},
    ])


def evaluate_pd(base, summary_rows):
    event_12m = base["is_event"].eq(1) & base["event_month"].le(12)
    eligible_12m = event_12m | base["obs_end"].ge(12)
    test = base["split"].eq("test")
    eligible_test = test & eligible_12m
    y = event_12m.loc[eligible_test].astype(np.int8).to_numpy()
    predicted = base.loc[eligible_test, "pd_12m"].to_numpy(dtype=np.float64)
    eligible_count = int(eligible_test.sum())
    event_count = int(y.sum())
    excluded_count = int((test & ~eligible_12m).sum())
    non_event_count = eligible_count - event_count
    observed_counts = {
        "eligible": eligible_count,
        "events": event_count,
        "excluded": excluded_count,
    }
    if observed_counts != EXPECTED_12M_TEST:
        raise ValueError(
            f"12-month test eligibility differs from expected endpoint: {observed_counts}."
        )
    if non_event_count != 30_213 or np.unique(y).size != 2:
        raise ValueError(
            f"12-month test endpoint has inconsistent classes: events={event_count}, "
            f"non_events={non_event_count}."
        )
    if not np.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("Eligible test pd_12m contains invalid probabilities.")

    auc = float(roc_auc_score(y, predicted))
    gini = 2.0 * auc - 1.0
    fpr, tpr, _ = roc_curve(y, predicted)
    ks = float(np.max(tpr - fpr))
    brier = float(brier_score_loss(y, predicted))
    observed_rate = float(y.mean())
    predicted_rate = float(predicted.mean())
    expected_events = float(predicted.sum())
    observed_expected = event_count / expected_events if expected_events else np.nan

    test_frame = base.loc[eligible_test, ["id", "sub_grade", "pd_12m"]].copy()
    test_frame["event_12m"] = y
    test_frame["pd_decile"] = pd.qcut(
        test_frame["pd_12m"].rank(method="first"), 10, labels=False
    ) + 1
    deciles = (test_frame.groupby("pd_decile", observed=True)
               .agg(
                   n=("id", "size"),
                   events=("event_12m", "sum"),
                   mean_predicted_pd=("pd_12m", "mean"),
                   observed_event_rate=("event_12m", "mean"),
                   min_pd=("pd_12m", "min"),
                   max_pd=("pd_12m", "max"),
               )
               .reset_index())
    deciles["non_events"] = deciles["n"] - deciles["events"]
    deciles["observed_expected_ratio"] = (
        deciles["events"] / (deciles["mean_predicted_pd"] * deciles["n"])
    )
    deciles["cohort"] = "Out-of-time test, eligible 12-month complete-follow-up subset"

    subgrade = (test_frame.groupby("sub_grade", dropna=False)
                .agg(
                    n=("id", "size"),
                    events=("event_12m", "sum"),
                    mean_predicted_pd=("pd_12m", "mean"),
                    observed_event_rate=("event_12m", "mean"),
                )
                .reset_index())
    subgrade["non_events"] = subgrade["n"] - subgrade["events"]
    subgrade["observed_expected_ratio"] = (
        subgrade["events"] / (subgrade["mean_predicted_pd"] * subgrade["n"])
    )
    subgrade["sample_size_note"] = np.where(
        subgrade["n"] >= MIN_SEGMENT_N,
        "n>=50; descriptive segment",
        "small segment; interpret cautiously",
    )
    subgrade["cohort"] = "Out-of-time test, eligible 12-month complete-follow-up subset"

    summary_rows.extend([
        {"section": "PD 12-month test", "metric": "eligible loans", "value": eligible_count, "notes": "Complete-follow-up subset; not full test cohort."},
        {"section": "PD 12-month test", "metric": "events / non-events", "value": f"{event_count} / {non_event_count}", "notes": "Event uses six-month charge-off proxy."},
        {"section": "PD 12-month test", "metric": "excluded test loans", "value": excluded_count, "notes": "Early censoring/prepayment or incomplete 12-month follow-up."},
        {"section": "PD 12-month test", "metric": "ROC-AUC", "value": auc, "notes": "Eligible out-of-time test only."},
        {"section": "PD 12-month test", "metric": "Gini", "value": gini, "notes": "2*AUC-1."},
        {"section": "PD 12-month test", "metric": "KS", "value": ks, "notes": "Maximum TPR-FPR from ROC curve."},
        {"section": "PD 12-month test", "metric": "Brier score", "value": brier, "notes": "Eligible out-of-time test only."},
        {"section": "PD 12-month test", "metric": "mean predicted PD", "value": predicted_rate, "notes": "Eligible out-of-time test only."},
        {"section": "PD 12-month test", "metric": "observed event rate", "value": observed_rate, "notes": "Eligible out-of-time test only."},
        {"section": "PD 12-month test", "metric": "observed / expected events", "value": observed_expected, "notes": f"Observed={event_count}; expected={expected_events:.4f}."},
    ])
    return deciles, subgrade, (fpr, tpr), summary_rows


def evaluate_lifetime_pd(base, summary_rows):
    rows = []
    test = base["split"].eq("test")
    for term in VALID_TERMS:
        event_in_term = base["is_event"].eq(1) & base["event_month"].le(term)
        eligible = event_in_term | base["obs_end"].ge(term)
        cohort = test & base["term_months"].eq(term)
        eligible_cohort = cohort & eligible
        y = event_in_term.loc[eligible_cohort].astype(np.int8).to_numpy()
        prediction = base.loc[eligible_cohort, "pd_lifetime"].to_numpy(dtype=np.float64)
        n = len(y)
        events = int(y.sum())
        non_events = n - events
        auc = float(roc_auc_score(y, prediction)) if np.unique(y).size == 2 else np.nan
        status = "computed" if np.isfinite(auc) else "unavailable: single outcome class"
        rows.append({
            "term_months": term,
            "eligible_test_loans": n,
            "events": events,
            "non_events": non_events,
            "excluded_test_loans": int((cohort & ~eligible).sum()),
            "roc_auc": auc,
            "gini": 2 * auc - 1 if np.isfinite(auc) else np.nan,
            "mean_predicted_lifetime_pd": float(prediction.mean()) if n else np.nan,
            "observed_event_rate": float(y.mean()) if n else np.nan,
            "auc_status": status,
        })
        summary_rows.append({
            "section": f"Lifetime PD test {term}m",
            "metric": "eligibility / events / non-events / excluded",
            "value": f"{n} / {events} / {non_events} / {int((cohort & ~eligible).sum())}",
            "notes": "Full-horizon complete-follow-up subset; test split only.",
        })
        summary_rows.append({
            "section": f"Lifetime PD test {term}m",
            "metric": "ROC-AUC",
            "value": auc if np.isfinite(auc) else "unavailable",
            "notes": status,
        })
    return pd.DataFrame(rows)


def validate_lgd(base, loans, summary_rows):
    chargeoffs = loans.loc[loans["loan_status"].eq("Charged Off")].copy()
    chargeoffs["ead_at_default_proxy"] = chargeoffs.apply(
        lgd_model.compute_ead_at_default, axis=1
    )
    recoveries = pd.to_numeric(chargeoffs["recoveries"], errors="coerce")
    fees = pd.to_numeric(chargeoffs["collection_recovery_fee"], errors="coerce")
    proxy_ead = pd.to_numeric(chargeoffs["ead_at_default_proxy"], errors="coerce")
    valid_target = (
        np.isfinite(proxy_ead)
        & proxy_ead.gt(0)
        & np.isfinite(recoveries)
        & np.isfinite(fees)
    )
    raw_lgd = pd.Series(np.nan, index=chargeoffs.index, dtype=np.float64)
    raw_lgd.loc[valid_target] = (
        1.0 - (recoveries.loc[valid_target] - fees.loc[valid_target])
        / proxy_ead.loc[valid_target]
    )
    expected_realised = raw_lgd.clip(0, 1)
    lgd_lookup = base.set_index("id")["lgd_realised"]
    actual_realised = chargeoffs["id"].map(lgd_lookup)
    if (expected_realised.notna() != actual_realised.notna()).any():
        raise ValueError("LGD realized-target availability differs from source reconstruction.")
    target_mask = expected_realised.notna()
    if not np.allclose(
        expected_realised.loc[target_mask].to_numpy(),
        actual_realised.loc[target_mask].to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Saved realized LGD differs from reconstructed clipped target.")

    ead_zero = int((np.isfinite(proxy_ead) & proxy_ead.le(0)).sum())
    clipped_below = int((valid_target & raw_lgd.lt(0)).sum())
    clipped_above = int((valid_target & raw_lgd.gt(1)).sum())
    target_outside_chargeoff = int(
        (base["lgd_realised"].notna() & ~base["loan_status"].eq("Charged Off")).sum()
    )
    if target_outside_chargeoff:
        raise ValueError("Realized LGD is populated outside Charged Off loans.")

    test_valid = (
        base["split"].eq("test")
        & base["loan_status"].eq("Charged Off")
        & base["lgd_realised"].notna()
    )
    d = base.loc[test_valid].copy()
    if len(d) != 5_776:
        raise ValueError(f"Expected 5,776 valid test LGD targets; found {len(d):,}.")
    observed = d["lgd_realised"].to_numpy(dtype=np.float64)
    predicted = d["lgd_predicted"].to_numpy(dtype=np.float64)
    if (not np.isfinite(observed).all() or not np.isfinite(predicted).all()
            or (observed < 0).any() or (observed > 1).any()
            or (predicted < 0).any() or (predicted > 1).any()):
        raise ValueError("LGD targets/predictions must be finite and within [0, 1].")
    metrics = {
        "n": len(d),
        "mean_realised": float(observed.mean()),
        "mean_predicted": float(predicted.mean()),
        "bias_predicted_minus_realised": float(np.mean(predicted - observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
        "exact_one_share": float(np.mean(observed == 1.0)),
        "chargeoffs_without_target": int(
            (base["loan_status"].eq("Charged Off") & base["lgd_realised"].isna()).sum()
        ),
        "zero_ead_exclusions": ead_zero,
        "negative_raw_lgd_clipped": clipped_below,
        "above_one_raw_lgd_clipped": clipped_above,
    }

    segment_rows = []
    for column, label in (("term_months", "term"), ("sub_grade", "sub_grade")):
        grouped = d.groupby(column, dropna=False)
        for segment, group in grouped:
            y = group["lgd_realised"].to_numpy(dtype=np.float64)
            p = group["lgd_predicted"].to_numpy(dtype=np.float64)
            segment_rows.append({
                "segment_type": label,
                "segment": str(segment),
                "n": len(group),
                "mean_realised_lgd": float(y.mean()),
                "mean_predicted_lgd": float(p.mean()),
                "bias_predicted_minus_realised": float(np.mean(p - y)),
                "mae": float(mean_absolute_error(y, p)),
                "rmse": float(mean_squared_error(y, p) ** 0.5),
                "r2": float(r2_score(y, p)) if len(group) > 1 and np.var(y) > 0 else np.nan,
                "exact_one_share": float(np.mean(y == 1.0)),
                "sample_size_note": (
                    "n>=50; descriptive" if len(group) >= MIN_SEGMENT_N
                    else "small segment; interpret cautiously"
                ),
            })
    segments = pd.DataFrame(segment_rows)

    summary_rows.extend([
        {"section": "LGD test", "metric": "valid charged-off targets", "value": metrics["n"], "notes": "Test split only; conditional on observed proxy default."},
        {"section": "LGD test", "metric": "mean realized / predicted", "value": f"{metrics['mean_realised']:.6f} / {metrics['mean_predicted']:.6f}", "notes": "LGD fractions."},
        {"section": "LGD test", "metric": "bias predicted - realized", "value": metrics["bias_predicted_minus_realised"], "notes": "Constant training-mean LGD baseline."},
        {"section": "LGD test", "metric": "MAE / RMSE / R2", "value": f"{metrics['mae']:.6f} / {metrics['rmse']:.6f} / {metrics['r2']:.6f}", "notes": "MAE/RMSE are LGD-fraction errors."},
        {"section": "LGD target", "metric": "Charged Off without valid target", "value": metrics["chargeoffs_without_target"], "notes": "Zero-EAD/invalid-target exclusions."},
        {"section": "LGD target", "metric": "EAD<=0 exclusions / raw LGD<0 clipped / >1 clipped", "value": f"{ead_zero} / {clipped_below} / {clipped_above}", "notes": "Recomputed from the existing six-month scheduled-balance proxy."},
        {"section": "LGD target", "metric": "exact-1 target share on test", "value": metrics["exact_one_share"], "notes": "Recovery maturity/vintage shift may affect comparability."},
    ])
    return metrics, segments, summary_rows


def validate_monthly_input(frame, meta, label, value_column, summary_rows):
    require_columns(frame, ["id", "split", "term_months", "month", value_column], label)
    if frame["id"].isna().any() or frame.duplicated(["id", "month"]).any():
        raise ValueError(f"{label} has null IDs or duplicate (id, month) keys.")
    expected_rows = int(meta["term_months"].sum())
    if len(frame) != expected_rows:
        raise ValueError(f"{label} has {len(frame):,} rows; expected {expected_rows:,}.")
    id_summary = frame.groupby("id", sort=False).agg(
        row_count=("month", "size"),
        min_month=("month", "min"),
        max_month=("month", "max"),
        unique_months=("month", "nunique"),
        split_count=("split", "nunique"),
        term_count=("term_months", "nunique"),
        split=("split", "first"),
        term_months=("term_months", "first"),
    )
    expected_meta = meta.set_index("id").loc[id_summary.index]
    if (id_summary["row_count"].ne(expected_meta["term_months"]).any()
            or id_summary["min_month"].ne(1).any()
            or id_summary["max_month"].ne(expected_meta["term_months"]).any()
            or id_summary["unique_months"].ne(expected_meta["term_months"]).any()
            or id_summary["split_count"].ne(1).any()
            or id_summary["term_count"].ne(1).any()
            or not id_summary["split"].eq(expected_meta["split"]).all()
            or not id_summary["term_months"].eq(expected_meta["term_months"]).all()):
        raise ValueError(f"{label} loan-month coverage/split/term is inconsistent.")
    assert_same_ids(id_summary.reset_index(), pd.Index(meta["id"]), label)
    return id_summary


def validate_monthly_schedules(meta, loans, summary_rows):
    hazards_path = PROCESSED / "pd_monthly_hazards.parquet"
    ead_path = PROCESSED / "ead_schedules.parquet"
    hazard_diagnostics = []
    ead_diagnostics = []
    max_pd12_error = 0.0
    max_lifetime_error = 0.0

    funding = loans.set_index("id")[["funded_amnt", "int_rate"]]
    pd_lookup = meta.set_index("id")[["pd_12m", "pd_lifetime"]]

    for term in VALID_TERMS:
        hazards = pd.read_parquet(
            hazards_path,
            columns=["id", "split", "term_months", "month", "hazard"],
            filters=[("term_months", "==", term)],
        )
        ead = pd.read_parquet(
            ead_path,
            columns=["id", "split", "term_months", "month", "ead"],
            filters=[("term_months", "==", term)],
        )
        h_summary = validate_monthly_input(hazards, meta.loc[meta.term_months.eq(term)],
                                           "Monthly hazards", "hazard", summary_rows)
        e_summary = validate_monthly_input(ead, meta.loc[meta.term_months.eq(term)],
                                           "EAD schedule", "ead", summary_rows)
        if (len(h_summary.index.difference(e_summary.index))
            or len(e_summary.index.difference(h_summary.index))):
            raise ValueError(f"Term {term}: hazard and EAD loan IDs do not align.")

        h = pd.to_numeric(hazards["hazard"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(h).all() or (h < 0).any() or (h > 1).any():
            raise ValueError(f"Term {term}: hazards must be finite in [0, 1].")
        hazards = hazards.assign(hazard_value=h).sort_values(
            ["id", "month"], kind="stable"
        ).reset_index(drop=True)
        safe_hazard = np.clip(hazards["hazard_value"].to_numpy(),
                              0.0, np.nextafter(1.0, 0.0))
        grouped_hazard = pd.Series(safe_hazard).groupby(
            hazards["id"], sort=False
        )
        survival_after = (1.0 - pd.Series(safe_hazard)).groupby(
            hazards["id"], sort=False
        ).cumprod()
        survival_before = survival_after.groupby(
            hazards["id"], sort=False
        ).shift(1, fill_value=1.0)
        marginal = safe_hazard * survival_before.to_numpy(dtype=np.float64)
        if (not np.isfinite(marginal).all()
                or (marginal < -PD_TOLERANCE).any()
                or (marginal > 1 + PD_TOLERANCE).any()):
            raise ValueError(f"Term {term}: marginal PDs are invalid.")
        monthly_pd = hazards[["id", "month"]].copy()
        monthly_pd["marginal_pd"] = marginal
        reconstructed_lifetime = monthly_pd.groupby("id", sort=False).marginal_pd.sum()
        reconstructed_pd12 = (monthly_pd.loc[monthly_pd.month.le(12)]
                              .groupby("id", sort=False).marginal_pd.sum())
        references = pd_lookup.loc[reconstructed_lifetime.index]
        lifetime_error = np.abs(
            reconstructed_lifetime.to_numpy() - references.pd_lifetime.to_numpy()
        )
        pd12_error = np.abs(
            reconstructed_pd12.to_numpy() - references.pd_12m.to_numpy()
        )
        max_lifetime_error = max(max_lifetime_error, float(lifetime_error.max()))
        max_pd12_error = max(max_pd12_error, float(pd12_error.max()))
        if (lifetime_error > PD_TOLERANCE).any() or (pd12_error > PD_TOLERANCE).any():
            raise ValueError(
                f"Term {term}: hazard-derived PD does not reconcile; "
                f"max12={pd12_error.max():.3e}, max_lifetime={lifetime_error.max():.3e}."
            )

        e = pd.to_numeric(ead["ead"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(e).all() or (e < 0).any():
            raise ValueError(f"Term {term}: EAD must be finite and nonnegative.")
        ead = ead.assign(ead_value=e).sort_values(
            ["id", "month"], kind="stable"
        ).reset_index(drop=True)
        if not hazards[["id", "month"]].equals(ead[["id", "month"]]):
            raise ValueError(f"Term {term}: hazard and EAD monthly keys differ.")

        ead_array = ead["ead_value"].to_numpy(dtype=np.float64)
        id_array = ead["id"].to_numpy()
        month_array = ead["month"].to_numpy(dtype=np.int64)
        same_loan = id_array[1:] == id_array[:-1]
        differences = np.diff(ead_array)
        if (differences[same_loan] > EAD_TOLERANCE).any():
            raise ValueError(f"Term {term}: contractual EAD increases during amortization.")

        start_mask = month_array == 1
        start_principal = funding.reindex(id_array[start_mask])["funded_amnt"].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(ead_array[start_mask], start_principal, rtol=0, atol=1e-8):
            raise ValueError(f"Term {term}: month-1 contractual EAD differs from funded amount.")

        final_mask = month_array == term
        final_ids = id_array[final_mask]
        final_start_balance = ead_array[final_mask]
        final_features = funding.reindex(final_ids)
        principal = final_features["funded_amnt"].to_numpy(dtype=np.float64)
        monthly_rate = final_features["int_rate"].to_numpy(dtype=np.float64) / 1200.0
        zero_rate = monthly_rate == 0
        payment = np.empty(len(final_ids), dtype=np.float64)
        payment[zero_rate] = principal[zero_rate] / term
        positive_rate = ~zero_rate
        payment[positive_rate] = (
            principal[positive_rate] * monthly_rate[positive_rate]
            / -np.expm1(-term * np.log1p(monthly_rate[positive_rate]))
        )
        residual = final_start_balance * (1 + monthly_rate) - payment
        if not np.allclose(residual, 0, rtol=0, atol=EAD_TOLERANCE):
            raise ValueError(
                f"Term {term}: final contractual payment leaves an invalid residual."
            )
        h_split = hazards.drop_duplicates("id").set_index("id")["split"]
        e_split = ead.drop_duplicates("id").set_index("id")["split"]
        if not h_split.reindex(e_split.index).eq(e_split).all():
            raise ValueError(f"Term {term}: hazard and EAD splits disagree.")
        hazard_diagnostics.append({
            "term_months": term,
            "loans": len(h_summary),
            "loan_month_rows": len(hazards),
            "hazard_min": float(h.min()),
            "hazard_max": float(h.max()),
            "max_pd12_reconciliation_error": float(pd12_error.max()),
            "max_lifetime_reconciliation_error": float(lifetime_error.max()),
        })
        ead_diagnostics.append({
            "term_months": term,
            "loans": len(e_summary),
            "loan_month_rows": len(ead),
            "ead_min": float(e.min()),
            "ead_max": float(e.max()),
            "max_month1_principal_error": float(np.max(np.abs(ead_array[start_mask] - start_principal))),
            "max_final_balance_residual": float(np.max(np.abs(residual))),
            "monotonic": True,
        })

    add_summary(summary_rows, "PD reconstruction", "maximum 12m absolute error", max_pd12_error,
                "Monthly hazards converted to marginal PD with survival.")
    add_summary(summary_rows, "PD reconstruction", "maximum lifetime absolute error", max_lifetime_error,
                "Monthly hazards converted to marginal PD with survival.")
    return pd.DataFrame(hazard_diagnostics), pd.DataFrame(ead_diagnostics), summary_rows


def validate_loan_status_and_endpoint(base, summary_rows):
    if not base["loan_status"].isin(["Fully Paid", "Charged Off", "Current"]).all():
        raise ValueError("Unexpected loan statuses in PD scores.")
    if not base["term_months"].isin(VALID_TERMS).all():
        raise ValueError("Unexpected contractual term in PD scores.")
    if not base["default"].eq(base["loan_status"].eq("Charged Off").astype(int)).all():
        raise ValueError("PD default indicator differs from loan_status.")
    last_payment = pd.to_numeric(base["months_to_last_pymnt"], errors="coerce")
    seasoned = pd.to_numeric(base["months_seasoned"], errors="coerce")
    term = pd.to_numeric(base["term_months"], errors="coerce")
    status = base["loan_status"]
    charged = status.eq("Charged Off")
    paid = status.eq("Fully Paid")
    event_month = last_payment + 6
    expected_event_month = event_month.where(charged)
    actual_event_month = pd.to_numeric(base["event_month"], errors="coerce")
    if not np.allclose(
        actual_event_month.to_numpy(dtype=np.float64),
        expected_event_month.to_numpy(dtype=np.float64),
        rtol=0, atol=1e-9, equal_nan=True,
    ):
        raise ValueError("event_month differs from last-payment month plus six-month proxy.")
    raw_end = np.where(charged, event_month, np.where(paid, last_payment, seasoned))
    expected_obs_end = np.floor(np.minimum(np.minimum(raw_end, seasoned), term))
    if not np.allclose(
        base["obs_end"].to_numpy(dtype=np.float64), expected_obs_end,
        rtol=0, atol=1e-9, equal_nan=True,
    ):
        raise ValueError("obs_end differs from pd.py censoring/term rule.")
    expected_is_event = (
        charged & event_month.notna() & event_month.ge(1)
        & event_month.le(seasoned) & event_month.le(term)
    ).astype(np.int8)
    if not base["is_event"].astype(np.int8).eq(expected_is_event).all():
        raise ValueError("is_event differs from pd.py event-observation rule.")
    if not base["obs_end"].between(1, base["term_months"]).all():
        raise ValueError("obs_end is outside the contractual term.")
    add_summary(summary_rows, "PD endpoint", "event/obs_end reconstruction", "passed",
                "Six-month charge-off lag; fully-paid/current censoring, snapshot, and term caps reproduced.")
    return summary_rows


def calculate_pd_validation(base):
    event_12m = base["is_event"].eq(1) & base["event_month"].le(12)
    eligible_12m = event_12m | base["obs_end"].ge(12)
    test = base["split"].eq("test")
    eligible_test = test & eligible_12m
    test_frame = base.loc[eligible_test].copy()
    y = event_12m.loc[eligible_test].astype(np.int8).to_numpy()
    predicted = test_frame["pd_12m"].to_numpy(dtype=np.float64)
    counts = {
        "eligible": len(test_frame),
        "events": int(y.sum()),
        "non_events": int(len(y) - y.sum()),
        "excluded": int((test & ~eligible_12m).sum()),
    }
    if (counts["eligible"] != 30_827 or counts["events"] != 614
            or counts["non_events"] != 30_213 or counts["excluded"] != 4_401):
        raise ValueError(f"12-month endpoint count check failed: {counts}")
    if not np.isfinite(predicted).all() or (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("Eligible test pd_12m has invalid probability values.")

    auc = float(roc_auc_score(y, predicted))
    gini = 2 * auc - 1
    fpr, tpr, thresholds = roc_curve(y, predicted)
    ks_index = int(np.argmax(tpr - fpr))
    ks = float((tpr - fpr)[ks_index])
    brier = float(brier_score_loss(y, predicted))
    mean_predicted = float(predicted.mean())
    observed_rate = float(y.mean())
    expected_events = float(predicted.sum())
    o_e = float(y.sum() / expected_events)

    test_frame["event_12m"] = y
    test_frame["pd_decile"] = pd.qcut(
        test_frame["pd_12m"].rank(method="first"), 10, labels=False
    ) + 1
    deciles = (test_frame.groupby("pd_decile", observed=True)
               .agg(
                   n=("id", "size"),
                   events=("event_12m", "sum"),
                   mean_predicted_pd=("pd_12m", "mean"),
                   observed_event_rate=("event_12m", "mean"),
                   min_predicted_pd=("pd_12m", "min"),
                   max_predicted_pd=("pd_12m", "max"),
               )
               .reset_index())
    deciles["non_events"] = deciles["n"] - deciles["events"]
    deciles["expected_events"] = deciles["mean_predicted_pd"] * deciles["n"]
    deciles["observed_expected_ratio"] = deciles["events"] / deciles["expected_events"]
    deciles["cohort"] = "Out-of-time test; eligible complete-follow-up subset"

    subgrades = (test_frame.groupby("sub_grade", dropna=False)
                 .agg(
                     n=("id", "size"),
                     events=("event_12m", "sum"),
                     mean_predicted_pd=("pd_12m", "mean"),
                     observed_event_rate=("event_12m", "mean"),
                 )
                 .reset_index())
    subgrades["non_events"] = subgrades["n"] - subgrades["events"]
    subgrades["expected_events"] = subgrades["mean_predicted_pd"] * subgrades["n"]
    subgrades["observed_expected_ratio"] = subgrades["events"] / subgrades["expected_events"]
    subgrades["sample_size_note"] = np.where(
        subgrades["n"] >= MIN_SEGMENT_N,
        "n>=50; descriptive",
        "small sample; interpret cautiously",
    )
    subgrades["cohort"] = "Out-of-time test; eligible complete-follow-up subset"

    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}
    pd_metrics = {
        "eligible_test_loans": counts["eligible"],
        "events": counts["events"],
        "non_events": counts["non_events"],
        "excluded_test_loans": counts["excluded"],
        "auc": auc,
        "gini": float(gini),
        "ks": ks,
        "ks_threshold": float(thresholds[ks_index]),
        "brier": brier,
        "mean_predicted_pd": mean_predicted,
        "observed_event_rate": observed_rate,
        "expected_events": expected_events,
        "observed_expected_ratio": o_e,
    }
    return pd_metrics, deciles, subgrades, roc_data


def calculate_lifetime_pd_validation(base):
    rows = []
    for term in VALID_TERMS:
        event_within_term = base["is_event"].eq(1) & base["event_month"].le(term)
        eligible = event_within_term | base["obs_end"].ge(term)
        cohort = base["split"].eq("test") & base["term_months"].eq(term)
        selected = cohort & eligible
        outcomes = event_within_term.loc[selected].astype(np.int8).to_numpy()
        predictions = base.loc[selected, "pd_lifetime"].to_numpy(dtype=np.float64)
        if (not np.isfinite(predictions).all() or (predictions < 0).any()
                or (predictions > 1).any()):
            raise ValueError(f"Term {term}: eligible lifetime PD values are invalid.")
        event_count = int(outcomes.sum())
        non_event_count = int(len(outcomes) - event_count)
        auc = (
            float(roc_auc_score(outcomes, predictions))
            if len(outcomes) > 0 and np.unique(outcomes).size == 2
            else np.nan
        )
        rows.append({
            "term_months": term,
            "eligible_test_loans": len(outcomes),
            "events": event_count,
            "non_events": non_event_count,
            "excluded_test_loans": int((cohort & ~eligible).sum()),
            "roc_auc": auc,
            "gini": 2 * auc - 1 if np.isfinite(auc) else np.nan,
            "auc_status": "computed" if np.isfinite(auc) else "unavailable: single class",
        })
    lifetime = pd.DataFrame(rows)
    expected = {
        36: (10_456, 3_038, 7_418, 13_913),
        60: (2_724, 2_724, 0, 8_135),
    }
    for _, row in lifetime.iterrows():
        expected_row = expected[int(row.term_months)]
        actual_row = (
            int(row.eligible_test_loans), int(row.events), int(row.non_events),
            int(row.excluded_test_loans),
        )
        if actual_row != expected_row:
            raise ValueError(
                f"Term {int(row.term_months)} lifetime eligibility differs: {actual_row}."
            )
    return lifetime


def calculate_lgd_segments(base):
    target_available = base["lgd_realised"].notna()
    non_default_target = target_available & ~base["loan_status"].eq("Charged Off")
    if non_default_target.any():
        raise ValueError("Realized LGD targets exist outside Charged Off loans.")
    test_default = (
        base["split"].eq("test")
        & base["loan_status"].eq("Charged Off")
        & target_available
    )
    test = base.loc[test_default].copy()
    if len(test) != 5_776:
        raise ValueError(f"Expected 5,776 valid test LGD targets; found {len(test):,}.")
    y = test["lgd_realised"].to_numpy(dtype=np.float64)
    p = test["lgd_predicted"].to_numpy(dtype=np.float64)
    if (not np.isfinite(y).all() or not np.isfinite(p).all()
            or (y < 0).any() or (y > 1).any() or (p < 0).any() or (p > 1).any()):
        raise ValueError("LGD test targets/predictions must be finite within [0, 1].")
    overall = {
        "n": len(test),
        "mean_realised": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "bias": float(np.mean(p - y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "exact_one_share": float(np.mean(y == 1.0)),
    }
    segments = []
    for column, label in (("term_months", "term"), ("sub_grade", "sub_grade")):
        for segment, group in test.groupby(column, dropna=False):
            truth = group["lgd_realised"].to_numpy(dtype=np.float64)
            prediction = group["lgd_predicted"].to_numpy(dtype=np.float64)
            segments.append({
                "segment_type": label,
                "segment": str(segment),
                "n": len(group),
                "mean_realised_lgd": float(truth.mean()),
                "mean_predicted_lgd": float(prediction.mean()),
                "bias_predicted_minus_realised": float(np.mean(prediction - truth)),
                "mae": float(mean_absolute_error(truth, prediction)),
                "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
                "r2": float(r2_score(truth, prediction)) if len(group) > 1 and np.var(truth) > 0 else np.nan,
                "exact_one_share": float(np.mean(truth == 1.0)),
                "sample_size_note": (
                    "n>=50; descriptive" if len(group) >= MIN_SEGMENT_N
                    else "small sample; interpret cautiously"
                ),
            })
    return overall, pd.DataFrame(segments), test


def validate_lgd_target_construction(base, summary_rows):
    chargeoffs = base.loc[base["loan_status"].eq("Charged Off")].copy()
    chargeoffs["ead_at_default_proxy"] = chargeoffs.apply(
        lgd_model.compute_ead_at_default, axis=1
    )
    recoveries = pd.to_numeric(chargeoffs["recoveries"], errors="coerce")
    fees = pd.to_numeric(chargeoffs["collection_recovery_fee"], errors="coerce")
    proxy_ead = pd.to_numeric(chargeoffs["ead_at_default_proxy"], errors="coerce")
    valid = np.isfinite(proxy_ead) & proxy_ead.gt(0) & np.isfinite(recoveries) & np.isfinite(fees)
    raw = pd.Series(np.nan, index=chargeoffs.index, dtype=np.float64)
    raw.loc[valid] = 1 - (recoveries.loc[valid] - fees.loc[valid]) / proxy_ead.loc[valid]
    expected_realised = raw.clip(0, 1)
    actual_realised = chargeoffs["lgd_realised"]
    if (expected_realised.notna() != actual_realised.notna()).any():
        raise ValueError("LGD target missingness differs from source target reconstruction.")
    valid_target = expected_realised.notna()
    if not np.allclose(
        expected_realised.loc[valid_target].to_numpy(),
        actual_realised.loc[valid_target].to_numpy(dtype=np.float64),
        rtol=0, atol=1e-10,
    ):
        raise ValueError("LGD realized values differ from clipped source reconstruction.")
    excluded = int((~valid).sum())
    clipped_below = int((valid & raw.lt(0)).sum())
    clipped_above = int((valid & raw.gt(1)).sum())
    add_summary(summary_rows, "LGD target", "charged-off target exclusions", excluded,
                "Invalid/zero EAD or invalid recoveries/fees in the scheduled-balance proxy.")
    add_summary(summary_rows, "LGD target", "raw LGD below zero clipped", clipped_below,
                "Reconstructed from recoveries, fees, and the existing LGD EAD proxy.")
    add_summary(summary_rows, "LGD target", "raw LGD above one clipped", clipped_above,
                "Reconstructed from recoveries, fees, and the existing LGD EAD proxy.")
    return {
        "charged_off": len(chargeoffs),
        "valid_targets": int(valid_target.sum()),
        "excluded_targets": excluded,
        "raw_below_zero_clipped": clipped_below,
        "raw_above_one_clipped": clipped_above,
    }


def validate_ead_schedules(meta, loans, summary_rows):
    diagnostics = []
    funding = loans.set_index("id")[["funded_amnt", "int_rate"]]
    for term in VALID_TERMS:
        schedule = pd.read_parquet(
            EAD_IN,
            columns=["id", "split", "term_months", "month", "ead"],
            filters=[("term_months", "==", term)],
        )
        expected_meta = meta.loc[meta["term_months"].eq(term)].set_index("id")
        if schedule.duplicated(["id", "month"]).any():
            raise ValueError(f"Term {term} EAD has duplicate (id, month) keys.")
        id_stats = schedule.groupby("id", sort=False).agg(
            rows=("month", "size"), min_month=("month", "min"),
            max_month=("month", "max"), unique_months=("month", "nunique"),
            split_count=("split", "nunique"), term_count=("term_months", "nunique"),
            split=("split", "first"), term_months=("term_months", "first"),
        )
        assert_same_ids(id_stats.reset_index(), pd.Index(expected_meta.index), f"{term}m EAD")
        if (id_stats["rows"].ne(term).any() or id_stats["min_month"].ne(1).any()
                or id_stats["max_month"].ne(term).any()
                or id_stats["unique_months"].ne(term).any()
                or id_stats["split_count"].ne(1).any()
                or id_stats["term_count"].ne(1).any()
                or not id_stats["split"].eq(expected_meta["split"]).all()
                or not id_stats["term_months"].eq(term).all()):
            raise ValueError(f"Term {term} EAD month coverage/split/term invalid.")
        balances = pd.to_numeric(schedule["ead"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(balances).all() or (balances < 0).any():
            raise ValueError(f"Term {term} EAD contains invalid balances.")

        ids = schedule["id"].to_numpy()
        months = schedule["month"].to_numpy(dtype=np.int64)
        order = np.lexsort((months, ids))
        sorted_ids = ids[order]
        sorted_months = months[order]
        sorted_balances = balances[order]
        same_id = sorted_ids[1:] == sorted_ids[:-1]
        if (np.diff(sorted_balances)[same_id] > EAD_TOLERANCE).any():
            raise ValueError(f"Term {term} EAD is not nonincreasing by month.")
        first_mask = sorted_months == 1
        funded = funding.reindex(sorted_ids[first_mask])["funded_amnt"].to_numpy(float)
        month1_error = float(np.max(np.abs(sorted_balances[first_mask] - funded)))
        if month1_error > EAD_TOLERANCE:
            raise ValueError(f"Term {term} month-1 EAD differs from funded amount.")

        final_mask = sorted_months == term
        final_features = funding.reindex(sorted_ids[final_mask])
        principal = final_features["funded_amnt"].to_numpy(dtype=np.float64)
        monthly_rate = final_features["int_rate"].to_numpy(dtype=np.float64) / 1200
        zero = monthly_rate == 0
        payment = np.empty(len(principal), dtype=np.float64)
        payment[zero] = principal[zero] / term
        positive = ~zero
        payment[positive] = (
            principal[positive] * monthly_rate[positive]
            / -np.expm1(-term * np.log1p(monthly_rate[positive]))
        )
        final_residual = sorted_balances[final_mask] * (1 + monthly_rate) - payment
        max_final_residual = float(np.max(np.abs(final_residual)))
        if max_final_residual > EAD_TOLERANCE:
            raise ValueError(f"Term {term} EAD final payment residual is invalid.")
        diagnostics.append({
            "term_months": term,
            "loans": len(id_stats),
            "rows": len(schedule),
            "minimum_ead": float(balances.min()),
            "maximum_ead": float(balances.max()),
            "month1_principal_max_abs_error": month1_error,
            "final_payment_max_abs_residual": max_final_residual,
            "nonincreasing": True,
        })
    add_summary(summary_rows, "EAD", "contractual schedule consistency", "passed",
                "Unique keys, full month coverage, month-1 principal, monotonic amortization, final-payment residual.")
    return pd.DataFrame(diagnostics), summary_rows


def validate_ecl(base, summary_rows):
    ecl12 = base["ecl_12m"].to_numpy(dtype=np.float64)
    lifetime = base["ecl_lifetime"].to_numpy(dtype=np.float64)
    if not np.isfinite(ecl12).all() or not np.isfinite(lifetime).all():
        raise ValueError("ECL aggregates contain non-finite values.")
    if (ecl12 < 0).any() or (lifetime < 0).any() or (ecl12 > lifetime + ECL_TOLERANCE).any():
        raise ValueError("ECL nonnegativity or 12-month/lifetime ordering failed.")
    pd12_error = np.abs(base["pd_12m_reconstructed"] - base["pd_12m"])
    lifetime_error = np.abs(base["pd_lifetime_reconstructed"] - base["pd_lifetime"])
    if pd12_error.max() > 1e-6 or lifetime_error.max() > 1e-6:
        raise ValueError("Saved ECL PD reconstruction does not match saved PD scores.")
    current = base["loan_status"].eq("Current")
    stage1 = current & base["synthetic_stage"].eq("Synthetic Stage 1")
    stage2 = current & base["synthetic_stage"].eq("Synthetic Stage 2")
    selected = base["ecl_selected_synthetic"].notna()
    if (selected & ~(stage1 | stage2)).any():
        raise ValueError("Selected synthetic ECL is present outside staged Current loans.")
    if not base.loc[stage1, "ecl_selected_synthetic"].eq(base.loc[stage1, "ecl_12m"]).all():
        raise ValueError("Selected Stage 1 ECL does not equal ecl_12m.")
    if not base.loc[stage2, "ecl_selected_synthetic"].eq(base.loc[stage2, "ecl_lifetime"]).all():
        raise ValueError("Selected Stage 2 ECL does not equal ecl_lifetime.")
    add_summary(summary_rows, "ECL", "arithmetic checks", "passed",
                "PD reconstruction, nonnegative values, 12m <= lifetime, selected ECL only for Current synthetic stages.")
    return summary_rows


def validate_rwa(base, summary_rows):
    current = base["loan_status"].eq("Current")
    expected_current = int(current.sum())
    if expected_current != 1_789:
        raise ValueError(f"Expected 1,789 Current loans; found {expected_current:,}.")
    if not base["included_in_current_portfolio"].eq(current).all():
        raise ValueError("RWA inclusion flags do not select exactly Current loans.")
    if base.loc[~current, "rwa_illustrative"].notna().any():
        raise ValueError("Historical/non-Current loans have populated RWA.")
    if not base.loc[current, "rwa_illustrative"].notna().all():
        raise ValueError("Current portfolio RWA is missing.")
    exposure = pd.to_numeric(base.loc[current, "out_prncp"], errors="coerce")
    baseline_rwa = base.loc[current, "rwa_illustrative"].to_numpy(dtype=np.float64)
    if not np.isfinite(exposure.to_numpy(dtype=np.float64)).all() or (exposure < 0).any():
        raise ValueError("Current out_prncp exposure contains invalid values.")
    if not np.allclose(baseline_rwa, exposure.to_numpy(dtype=np.float64), rtol=0, atol=0.01):
        raise ValueError("Illustrative baseline RWA does not reconcile to Current exposure.")
    exposure_total = float(exposure.sum())
    if abs(exposure_total - 4_413_625.59) >= 0.01:
        raise ValueError(f"Current exposure sum differs from expected: {exposure_total:.2f}.")
    add_summary(summary_rows, "RWA", "Current exposure/RWA arithmetic", exposure_total,
                "Hypothetical 100% weight; arithmetic check only, not regulatory validation.")
    return int(current.sum()), int((current & exposure.eq(0)).sum()), exposure_total


def make_plots(roc_data, deciles):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(roc_data["fpr"], roc_data["tpr"], label=f"Eligible test ROC (AUC={roc_data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-discrimination")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="12-month PD ROC: out-of-time test, eligible follow-up subset")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROC_FIGURE_OUT, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(deciles["mean_predicted_pd"], deciles["observed_event_rate"],
            marker="o", label="Test PD deciles")
    for row in deciles.itertuples(index=False):
        ax.annotate(str(row.pd_decile),
                    (row.mean_predicted_pd, row.observed_event_rate),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean predicted 12-month PD", ylabel="Observed 12-month event rate",
           title="12-month PD calibration: out-of-time eligible test subset")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIGURE_OUT, dpi=160)
    plt.close(fig)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    base, pd_scores, loans, lgd, staging, ecl, rwa = load_inputs()
    missingness = validate_loan_level_values(base, summary_rows)
    for row in missingness.itertuples(index=False):
        add_summary(summary_rows, "Missingness", row.field, row.missing_count,
                    f"{row.missing_pct:.4%}; {row.interpretation}.")
    assert_same_ids(base, pd.Index(pd_scores["id"]), "Combined loan universe")
    add_summary(summary_rows, "Data integrity", "PD universe loans", len(base), "Authoritative ID universe.")
    add_summary(summary_rows, "Data integrity", "unique PD IDs", int(base["id"].nunique()), "All loan-level joins are one-to-one.")
    add_summary(summary_rows, "Data integrity", "train/test counts", str(base["split"].value_counts().to_dict()), "Expected 31,460 train and 35,228 out-of-time test.")
    add_summary(summary_rows, "Data integrity", "status counts", str(base["loan_status"].value_counts().to_dict()), "Fully Paid, Charged Off, and Current.")
    test_years = sorted(base.loc[base["split"].eq("test"), "issue_year"].dropna().unique().tolist())
    train_years = sorted(base.loc[base["split"].eq("train"), "issue_year"].dropna().unique().tolist())
    if test_years != [2014.0] or train_years != [2011.0, 2012.0, 2013.0]:
        raise ValueError(f"Unexpected OOT vintage split: train={train_years}, test={test_years}.")
    add_summary(summary_rows, "Data integrity", "issue-year cohorts", f"train={train_years}; test={test_years}", "2014 is the out-of-time test vintage.")

    validate_loan_status_and_endpoint(base, summary_rows)
    pd_metrics, deciles, subgrades, roc_data = calculate_pd_validation(base)
    for metric, value in pd_metrics.items():
        add_summary(summary_rows, "PD 12-month test", metric, value,
                    "Out-of-time eligible test subset only; complete-follow-up selection applies.")
    lifetime = calculate_lifetime_pd_validation(base)
    for row in lifetime.itertuples(index=False):
        add_summary(summary_rows, f"Lifetime PD test {row.term_months}m", "eligibility", row.eligible_test_loans,
                    f"events={row.events}; non_events={row.non_events}; excluded={row.excluded_test_loans}.")
        add_summary(summary_rows, f"Lifetime PD test {row.term_months}m", "ROC-AUC",
                    row.roc_auc if np.isfinite(row.roc_auc) else "unavailable",
                    row.auc_status)

    lgd_overall, lgd_segments, _ = calculate_lgd_segments(base)
    target_audit = validate_lgd_target_construction(base, summary_rows)
    for metric, value in lgd_overall.items():
        add_summary(summary_rows, "LGD test", metric, value,
                    "Test Charged Off loans with valid clipped realized targets only.")
    add_summary(summary_rows, "LGD test", "model interpretation", "constant training-mean baseline",
                "Not a differentiated or regulatory downturn LGD model.")

    # Check the existing loan-level EAD/hazard schedules without recomputing ECL.
    hazard_diagnostics, ead_diagnostics, summary_rows = validate_monthly_schedules(
        base, loans, summary_rows
    )
    summary_rows = validate_ecl(base, summary_rows)
    current_rwa_count, current_zero_count, current_exposure = validate_rwa(base, summary_rows)
    add_summary(summary_rows, "RWA", "Current included / zero balance", f"{current_rwa_count} / {current_zero_count}",
                "RWA output is an illustrative hypothetical 100% risk-weight proxy only.")

    summary_rows.extend([
        {"section": "Limitations", "metric": "loan selection", "value": "15% deterministic sample stratified by status/term/year/split", "notes": "May affect representativeness; no population weighting applied."},
        {"section": "Limitations", "metric": "PD endpoint", "value": "incomplete follow-up and early payoff", "notes": "Eligible test metrics condition on complete 12-month follow-up; not IPCW/competing-risk estimates."},
        {"section": "Limitations", "metric": "default timing", "value": "last payment + six months", "notes": "Proxy charge-off date, not verified default date."},
        {"section": "Limitations", "metric": "PD basis", "value": "origination-based estimates", "notes": "Not reporting-date PDs or regulatory long-run average PDs."},
        {"section": "Limitations", "metric": "LGD basis", "value": "historical recovery target, constant train-mean prediction", "notes": f"{target_audit['excluded_targets']} target exclusions; {target_audit['raw_below_zero_clipped']} below-zero raw targets clipped."},
        {"section": "Limitations", "metric": "EAD basis", "value": "contractual amortization schedule", "notes": "Not observed or regulatory EAD backtesting."},
        {"section": "Limitations", "metric": "ECL basis", "value": "synthetic stages and contractual-rate discount proxy", "notes": "Mathematical checks only; not empirical validation of IFRS 9 allowances."},
        {"section": "Limitations", "metric": "RWA basis", "value": "hypothetical Current exposure x assumed weight", "notes": "Not Basel/CRR regulatory capital validation."},
    ])

    deciles.to_csv(DECILES_OUT, index=False)
    subgrades.to_csv(SUBGRADE_OUT, index=False)
    lgd_segments.to_csv(LGD_SEGMENTS_OUT, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
    make_plots(roc_data, deciles)

    print("DATA INTEGRITY")
    print(f"  PD loans={len(base):,}; unique IDs={base['id'].nunique():,}; "
          f"splits={base['split'].value_counts().to_dict()}; "
          f"statuses={base['loan_status'].value_counts().to_dict()}")
    print("  Source status/term/split and saved-output ID joins: passed.")
    print("  Missingness in key PD-universe fields:")
    print(missingness.to_string(index=False))
    print("PD VALIDATION (eligible out-of-time 2014 test only)")
    print(f"  Eligible={pd_metrics['eligible_test_loans']:,}; events={pd_metrics['events']:,}; "
          f"non-events={pd_metrics['non_events']:,}; excluded={pd_metrics['excluded_test_loans']:,}")
    print(f"  AUC={pd_metrics['auc']:.4f}; Gini={pd_metrics['gini']:.4f}; "
          f"KS={pd_metrics['ks']:.4f}; Brier={pd_metrics['brier']:.5f}")
    print(f"  Mean PD={pd_metrics['mean_predicted_pd']:.2%}; "
          f"observed rate={pd_metrics['observed_event_rate']:.2%}; "
          f"O/E={pd_metrics['observed_expected_ratio']:.3f}")
    print("LIFETIME PD TEST ELIGIBILITY")
    print(lifetime.to_string(index=False))
    print("LGD VALIDATION (test Charged Off with valid target)")
    print(f"  n={lgd_overall['n']:,}; mean realized={lgd_overall['mean_realised']:.2%}; "
          f"mean predicted={lgd_overall['mean_predicted']:.2%}; "
          f"bias={lgd_overall['bias']:.4f}; MAE={lgd_overall['mae']:.4f}; "
          f"RMSE={lgd_overall['rmse']:.4f}; R2={lgd_overall['r2']:.4f}")
    print(f"  Target audit: {target_audit['charged_off']:,} charge-offs; "
          f"{target_audit['excluded_targets']:,} exclusions; "
          f"{target_audit['raw_below_zero_clipped']:,} raw LGDs below zero clipped.")
    print("EAD / ECL / RWA CHECKS")
    print("  EAD by term:")
    print(ead_diagnostics.to_string(index=False))
    print("  Hazard-to-PD reconciliation by term:")
    print(hazard_diagnostics.to_string(index=False))
    print(f"  ECL: total 12m={base['ecl_12m'].sum():,.2f}; "
          f"total lifetime={base['ecl_lifetime'].sum():,.2f}; "
          f"selected Current synthetic={base['ecl_selected_synthetic'].sum():,.2f}")
    print(f"  Illustrative RWA arithmetic: Current loans={current_rwa_count:,}; "
          f"zero balance={current_zero_count:,}; exposure={current_exposure:,.2f}.")
    print("LIMITATIONS")
    print("  Loan-status-stratified 15% sample; early censoring/prepayment; six-month "
          "charge-off proxy; origination PD; historical clipped LGD and target exclusions; "
          "contractual EAD; synthetic staging/non-EIR discounting; hypothetical RWA. "
          "No regulatory IFRS 9 or Basel/CRR validation is claimed.")
    print("OUTPUTS")
    for path in (SUMMARY_OUT, DECILES_OUT, SUBGRADE_OUT, LGD_SEGMENTS_OUT,
                 ROC_FIGURE_OUT, CALIBRATION_FIGURE_OUT):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected report output missing/empty: {path}")
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()