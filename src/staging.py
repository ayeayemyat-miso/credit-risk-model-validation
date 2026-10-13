"""Phase 5: observed status labels and an illustrative SICR scenario.

The synthetic Stage 1/2 labels are educational sensitivities, not observed
IFRS 9 classifications, a fitted reporting-date PD model, or evidence of IFRS 9
compliance. Origination lifetime PD is stressed only for Current loans using a
deterministic loan-ID hash assignment. The remaining-lifetime comparability
required for a genuine SICR assessment cannot be established from the inputs.

Run from the project root: python src/staging.py
Inputs: data/processed/loans_clean.parquet and pd_scores.parquet
Output: data/processed/staging.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOANS_IN = ROOT / "data" / "processed" / "loans_clean.parquet"
PD_IN = ROOT / "data" / "processed" / "pd_scores.parquet"
OUT = ROOT / "data" / "processed" / "staging.parquet"

EXPECTED_LOAN_COUNT = 66_688
SYNTHETIC_PD_MULTIPLIERS = (1.5, 2.0, 3.0, 4.0)
ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD = 3.0
SYNTHETIC_ID_HASH_KEY = "staging-scenario"

OBSERVED_CLASSIFICATION = {
    "Fully Paid": "Closed",
    "Charged Off": "Historical Stage 3 proxy",
    "Current": "Unassessed",
}


def load_and_join():
    """Join source status to all PD loans while retaining PD split and term."""
    loans = pd.read_parquet(LOANS_IN)
    pd_scores = pd.read_parquet(PD_IN)

    for frame, label in ((loans, "clean loan"), (pd_scores, "PD")):
        if "id" not in frame:
            raise ValueError(f"{label} file is missing required column 'id'.")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{label} loan IDs must be non-null and unique.")

    pd_columns = ["id", "split", "term_months", "pd_lifetime"]
    missing_pd = [column for column in pd_columns if column not in pd_scores.columns]
    if missing_pd:
        raise ValueError(f"PD scores are missing required columns: {missing_pd}")
    if not pd_scores["split"].isin(["train", "test"]).all():
        raise ValueError("PD split must contain only 'train' or 'test'.")

    source_columns = ["id", "loan_status", "term_months", "split"]
    missing_source = [column for column in source_columns if column not in loans.columns]
    if missing_source:
        raise ValueError(f"Clean loans are missing required columns: {missing_source}")
    source = loans[source_columns].rename(columns={
        "loan_status": "loan_status_source",
        "term_months": "term_months_source",
        "split": "split_source",
    })

    joined = pd_scores[pd_columns + (["loan_status"] if "loan_status" in pd_scores else [])].merge(
        source,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_ids = joined["_merge"].ne("both")
    if missing_ids.any():
        raise ValueError(
            f"{int(missing_ids.sum()):,} PD loans are missing from clean loan data."
        )
    if joined["split"].ne(joined["split_source"]).any():
        count = int(joined["split"].ne(joined["split_source"]).sum())
        raise ValueError(f"PD and source train/test splits disagree for {count:,} loans.")
    if joined["term_months"].ne(joined["term_months_source"]).any():
        count = int(joined["term_months"].ne(joined["term_months_source"]).sum())
        raise ValueError(f"PD and source terms disagree for {count:,} loans.")
    if "loan_status" in pd_scores:
        mismatch = joined["loan_status"].ne(joined["loan_status_source"])
        if mismatch.any():
            raise ValueError(
                f"PD and source loan statuses disagree for {int(mismatch.sum()):,} loans."
            )

    joined["loan_status"] = joined["loan_status_source"]
    joined = joined.drop(columns=[
        "loan_status_source", "term_months_source", "split_source", "_merge",
    ])
    if len(joined) != EXPECTED_LOAN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOAN_COUNT:,} PD loans; found {len(joined):,}."
        )
    if joined["id"].isna().any() or joined["id"].duplicated().any():
        raise AssertionError("Joined PD loan IDs must remain non-null and unique.")

    unexpected = sorted(set(joined["loan_status"].dropna())
                        .difference(OBSERVED_CLASSIFICATION))
    if joined["loan_status"].isna().any() or unexpected:
        raise ValueError(f"Unexpected or missing loan statuses: {unexpected}")
    return joined, pd_scores


def assign_synthetic_stages(joined):
    """Apply a deterministic, hypothetical lifetime-PD stress to Current loans."""
    multipliers = np.asarray(SYNTHETIC_PD_MULTIPLIERS, dtype=np.float64)
    if (multipliers.size < 2 or not np.isfinite(multipliers).all()
            or (multipliers < 1).any()):
        raise ValueError("Configure at least two finite synthetic multipliers >= 1.")
    if not np.isfinite(ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD) \
            or ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD <= 1:
        raise ValueError("Illustrative SICR threshold must be finite and greater than 1.")

    observed = joined["loan_status"].map(OBSERVED_CLASSIFICATION)
    origination_pd = pd.to_numeric(joined["pd_lifetime"], errors="coerce")
    synthetic_stage = np.full(len(joined), None, dtype=object)
    reason = np.full(len(joined), None, dtype=object)
    basis = np.full(len(joined), None, dtype=object)
    reporting_pd = np.full(len(joined), np.nan, dtype=np.float64)
    ratio = np.full(len(joined), np.nan, dtype=np.float64)

    closed = joined["loan_status"].eq("Fully Paid").to_numpy()
    charged_off = joined["loan_status"].eq("Charged Off").to_numpy()
    current = joined["loan_status"].eq("Current").to_numpy()

    reason[closed] = "Fully Paid; no longer an active loan exposure."
    basis[closed] = "Observed loan_status; classified Closed."
    reason[charged_off] = (
        "Charged Off is a historical credit-impaired proxy; reporting-date "
        "recognition/exposure is not established."
    )
    basis[charged_off] = (
        "Observed loan_status; historical Stage 3 proxy, not confirmed reporting-date Stage 3."
    )
    reason[current] = "Current loans have no observed reporting-date SICR evidence."
    basis[current] = (
        "Illustrative synthetic SICR scenario only; not observed IFRS 9 staging."
    )

    current_positions = np.flatnonzero(current)
    current_ids = joined.iloc[current_positions]["id"].astype(str)
    # Stable hashes allocate factors independently of default/status outcomes.
    hashes = pd.util.hash_pandas_object(
        current_ids,
        index=False,
        hash_key=SYNTHETIC_ID_HASH_KEY,
    ).to_numpy(dtype=np.uint64)
    assigned_multipliers = multipliers[hashes % multipliers.size]

    origination_values = origination_pd.to_numpy(dtype=np.float64)
    current_pd = origination_values[current_positions]
    valid_pd = np.isfinite(current_pd) & (current_pd > 0) & (current_pd <= 1)
    zero_pd = np.isfinite(current_pd) & (current_pd == 0)

    zero_positions = current_positions[zero_pd]
    reporting_pd[zero_positions] = 0.0
    reason[zero_positions] = (
        "Unassessable: zero origination lifetime PD makes a relative-PD ratio undefined."
    )
    invalid_positions = current_positions[~valid_pd & ~zero_pd]
    reason[invalid_positions] = (
        "Unassessable: origination lifetime PD is missing, non-finite, or outside [0, 1]."
    )

    eligible_positions = current_positions[valid_pd]
    eligible_multipliers = assigned_multipliers[valid_pd]
    eligible_origination_pd = current_pd[valid_pd]
    eligible_reporting_pd = np.minimum(
        eligible_origination_pd * eligible_multipliers, 1.0
    )
    eligible_ratio = eligible_reporting_pd / eligible_origination_pd
    synthetic_stage[eligible_positions] = np.where(
        eligible_ratio >= ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD,
        "Synthetic Stage 2",
        "Synthetic Stage 1",
    )
    reporting_pd[eligible_positions] = eligible_reporting_pd
    ratio[eligible_positions] = eligible_ratio

    for position, multiplier, pd_ratio in zip(
        eligible_positions, eligible_multipliers, eligible_ratio
    ):
        if pd_ratio >= ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD:
            reason[position] = (
                f"Synthetic lifetime-PD ratio {pd_ratio:.2f}x meets the "
                f"illustrative {ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD:.1f}x threshold."
            )
        else:
            reason[position] = (
                f"Synthetic lifetime-PD ratio {pd_ratio:.2f}x is below the "
                f"illustrative {ILLUSTRATIVE_SICR_PD_RATIO_THRESHOLD:.1f}x threshold."
            )
        basis[position] = (
            "Origination lifetime PD multiplied by the deterministic ID-hash factor "
            f"{multiplier:.1f}x and capped at 1.0; hypothetical only. The 3x threshold "
            "is illustrative, not IFRS 9/ECB-mandated; remaining-lifetime comparability "
            "is unverified."
        )

    result = pd.DataFrame({
        "id": joined["id"].to_numpy(),
        "split": joined["split"].to_numpy(),
        "term_months": joined["term_months"].to_numpy(),
        "loan_status": joined["loan_status"].to_numpy(),
        "observed_classification": observed.to_numpy(),
        "synthetic_stage": synthetic_stage,
        "sicr_reason": reason,
        "pd_origination_lifetime": origination_values,
        "pd_reporting_lifetime_synthetic": reporting_pd,
        "sicr_pd_ratio": ratio,
        "staging_basis": basis,
    })
    return result, assigned_multipliers, current_positions


def validate_output(result, pd_scores, current_positions):
    """Check population, status mapping, synthetic scope, and split integrity."""
    if len(result) != EXPECTED_LOAN_COUNT or result["id"].nunique() != EXPECTED_LOAN_COUNT:
        raise AssertionError(f"Output must contain {EXPECTED_LOAN_COUNT:,} unique PD loans.")
    if result["id"].isna().any() or result["id"].duplicated().any():
        raise AssertionError("Output IDs must be non-null and unique.")

    expected_ids = pd.Index(pd_scores["id"])
    actual_ids = pd.Index(result["id"])
    if len(expected_ids.difference(actual_ids)) or len(actual_ids.difference(expected_ids)):
        raise AssertionError("Output lost or added loan IDs relative to the PD universe.")
    expected_split = pd_scores.set_index("id")["split"].reindex(result["id"]).to_numpy()
    if not np.array_equal(result["split"].to_numpy(), expected_split):
        raise AssertionError("Output train/test splits differ from authoritative PD splits.")

    fully_paid = result["loan_status"].eq("Fully Paid")
    charged_off = result["loan_status"].eq("Charged Off")
    current = result["loan_status"].eq("Current")
    if not result.loc[fully_paid, "observed_classification"].eq("Closed").all():
        raise AssertionError("Fully Paid loans must be classified Closed.")
    if not result.loc[charged_off, "observed_classification"].eq(
        "Historical Stage 3 proxy"
    ).all():
        raise AssertionError("Charged Off loans must be historical Stage 3 proxies.")

    assigned = result["synthetic_stage"].notna()
    if (assigned & ~current).any():
        raise AssertionError("Synthetic stages may only be assigned to Current loans.")
    allowed_stages = {"Synthetic Stage 1", "Synthetic Stage 2"}
    if not set(result.loc[assigned, "synthetic_stage"]).issubset(allowed_stages):
        raise AssertionError("Unexpected synthetic stage label.")
    if not result.loc[~current, "synthetic_stage"].isna().all():
        raise AssertionError("Non-Current loans must not have synthetic stages.")

    ratios = result.loc[assigned, "sicr_pd_ratio"].to_numpy(dtype=np.float64)
    if not np.isfinite(ratios).all():
        raise AssertionError("Classified synthetic stages must have finite PD ratios.")
    expected_reporting_pd = result.loc[assigned, "pd_reporting_lifetime_synthetic"]
    if not np.isfinite(expected_reporting_pd.to_numpy(dtype=np.float64)).all():
        raise AssertionError("Classified synthetic stages must have finite reporting PDs.")
    if not expected_reporting_pd.between(0, 1).all():
        raise AssertionError("Synthetic reporting lifetime PDs must be in [0, 1].")
    if len(current_positions) != int(current.sum()):
        raise AssertionError("Current-loan scenario assignment count is inconsistent.")


def main():
    print("Loading authoritative PD universe and clean loan statuses ...")
    joined, pd_scores = load_and_join()
    result, assigned_multipliers, current_positions = assign_synthetic_stages(joined)
    validate_output(result, pd_scores, current_positions)

    print("\nObserved classification counts:")
    print(result["observed_classification"].value_counts(dropna=False).to_string())
    print("\nIllustrative synthetic stage counts for Current loans:")
    print(result.loc[result["loan_status"].eq("Current"), "synthetic_stage"]
          .value_counts(dropna=False).to_string())
    print("\nSynthetic factor assignment counts (Current only):")
    print(pd.Series(assigned_multipliers, name="multiplier")
          .value_counts().sort_index().to_string())
    print("\nMissingness and unassessable counts:")
    for column in ("pd_origination_lifetime", "pd_reporting_lifetime_synthetic",
                   "sicr_pd_ratio", "synthetic_stage"):
        missing = int(result[column].isna().sum())
        print(f"  {column}: {missing:,}/{len(result):,} ({missing / len(result):.2%}) missing")
    current_unassessable = result["loan_status"].eq("Current") & result["synthetic_stage"].isna()
    print(f"  Current unassessable: {int(current_unassessable.sum()):,}")

    print("\nValidation results:")
    print(f"  Unique PD IDs preserved: {result['id'].nunique():,}/{EXPECTED_LOAN_COUNT:,}")
    print(f"  Train/test counts preserved: {result['split'].value_counts().to_dict()}")
    print("  Fully Paid -> Closed: "
          f"{int(result.loc[result.loan_status.eq('Fully Paid'), 'observed_classification'].eq('Closed').sum()):,}/"
          f"{int(result.loan_status.eq('Fully Paid').sum()):,}")
    print("  Charged Off -> Historical Stage 3 proxy: "
          f"{int(result.loc[result.loan_status.eq('Charged Off'), 'observed_classification'].eq('Historical Stage 3 proxy').sum()):,}/"
          f"{int(result.loan_status.eq('Charged Off').sum()):,}")
    print(f"  Synthetic stages assigned only to Current: "
          f"{int((result.synthetic_stage.notna() & result.loan_status.eq('Current')).sum()):,}/"
          f"{int(result.synthetic_stage.notna().sum()):,}")
    print("  Reporting-PD horizon caveat: origination lifetime PD is compared with a "
          "hypothetical lifetime PD; remaining-lifetime comparability is not established.")
    print("  Synthetic Stage 1/2 labels are illustrative, not observed IFRS 9 compliance.")

    output_columns = [
        "id", "split", "term_months", "loan_status", "observed_classification",
        "synthetic_stage", "sicr_reason", "pd_origination_lifetime",
        "pd_reporting_lifetime_synthetic", "sicr_pd_ratio", "staging_basis",
    ]
    result[output_columns].to_parquet(OUT, index=False, compression="zstd")
    print(f"\nSaved {len(result):,} loans to {OUT}")


if __name__ == "__main__":
    main()