"""Illustrative portfolio RWA proxy; not a Basel/CRR calculation.

This educational scenario applies a hypothetical 100% risk weight to recorded
out_prncp for Current loans only. It does not determine a regulatory exposure
class, recognize collateral/guarantees, or establish a regulatory CCF. The
loan-level output and sensitivity totals are not regulatory capital results.

Run from the project root: python src/rwa.py
Inputs: data/processed/loans_clean.parquet and pd_scores.parquet
Optional diagnostic: data/processed/ecl_results.parquet
Output: data/processed/rwa_results.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LOANS_IN = PROCESSED / "loans_clean.parquet"
PD_IN = PROCESSED / "pd_scores.parquet"
ECL_IN = PROCESSED / "ecl_results.parquet"
OUT = PROCESSED / "rwa_results.parquet"

EXPECTED_LOAN_COUNT = 66_688
EXPECTED_CURRENT_EXPOSURE = 4_413_625.59
EXPOSURE_RECONCILIATION_TOLERANCE = 0.01
BASELINE_RISK_WEIGHT = 1.00
SENSITIVITY_RISK_WEIGHTS = (0.75, 1.00, 1.25, 1.50)

OUTPUT_COLUMNS = [
    "id", "split", "loan_status", "term_months", "purpose",
    "application_type", "out_prncp_snapshot", "exposure_basis",
    "illustrative_exposure_class", "risk_weight_assumption",
    "collateral_recognized", "ccf_assumption", "rwa_illustrative",
    "included_in_current_portfolio", "exclusion_reason",
    "calculation_basis", "assumption_label",
]

CURRENT_EXPOSURE_BASIS = (
    "Recorded out_prncp snapshot used as an educational on-balance-sheet "
    "exposure proxy; balance as-of date is unverified."
)
NONCURRENT_EXPOSURE_BASIS = (
    "Recorded out_prncp retained for transparency only; not included as live exposure."
)
COLLATERAL_UNAVAILABLE = (
    "Not assessed: eligible collateral/guarantee data are unavailable; "
    "no collateral recognition applied."
)
CURRENT_CCF_BASIS = (
    "Not applicable to this funded on-balance-sheet proxy; regulatory CCF undetermined."
)
CALCULATION_BASIS = (
    "Illustrative Current-portfolio exposure proxy: recorded out_prncp multiplied "
    "by a hypothetical risk-weight assumption; not Basel/CRR RWA."
)
CURRENT_ASSUMPTION_LABEL = (
    "Hypothetical educational 100% baseline risk weight; sensitivities at 75%, "
    "125%, and 150%; not an applicable or necessarily conservative Basel/CRR weight."
)
EXCLUDED_ASSUMPTION_LABEL = (
    "Historical/non-current loan excluded from the selected live portfolio; "
    "no risk weight assigned."
)


def require_unique_ids(frame, label):
    if "id" not in frame.columns:
        raise ValueError(f"{label} is missing required column 'id'.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{label} IDs must be non-null and unique.")


def require_same_ids(frame, expected_ids, label):
    require_unique_ids(frame, label)
    actual_ids = pd.Index(frame["id"])
    if len(actual_ids) != len(expected_ids):
        raise ValueError(
            f"{label} has {len(actual_ids):,} loans; expected {len(expected_ids):,}."
        )
    missing_ids = expected_ids.difference(actual_ids)
    extra_ids = actual_ids.difference(expected_ids)
    if len(missing_ids) or len(extra_ids):
        raise ValueError(
            f"{label} ID mismatch: {len(missing_ids):,} missing, "
            f"{len(extra_ids):,} extra."
        )


def load_current_portfolio():
    pd_scores = pd.read_parquet(PD_IN)
    if len(pd_scores) != EXPECTED_LOAN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOAN_COUNT:,} authoritative PD loans; "
            f"found {len(pd_scores):,}."
        )
    require_unique_ids(pd_scores, "PD scores")
    required_pd = {"id", "split", "term_months", "loan_status"}
    missing_pd = sorted(required_pd.difference(pd_scores.columns))
    if missing_pd:
        raise ValueError(f"PD scores are missing required columns: {missing_pd}")
    if not pd_scores["split"].isin(["train", "test"]).all():
        raise ValueError("PD split must contain only 'train' or 'test'.")

    loan_columns = [
        "id", "split", "term_months", "loan_status", "purpose",
        "application_type", "out_prncp",
    ]
    loans = pd.read_parquet(LOANS_IN, columns=loan_columns)
    require_unique_ids(loans, "Clean loans")
    source = loans.rename(columns={
        "split": "split_source",
        "term_months": "term_months_source",
        "loan_status": "loan_status_source",
    })
    universe = pd_scores[list(required_pd)].merge(
        source,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_source = universe["_merge"].ne("both")
    if missing_source.any():
        raise ValueError(
            f"{int(missing_source.sum()):,} PD loans are missing from clean loans."
        )
    for pd_column, source_column, label in (
        ("split", "split_source", "split"),
        ("term_months", "term_months_source", "term"),
        ("loan_status", "loan_status_source", "loan status"),
    ):
        mismatch = universe[pd_column].ne(universe[source_column])
        if mismatch.any():
            raise ValueError(
                f"PD/clean-loan {label} mismatch for {int(mismatch.sum()):,} loans."
            )
    universe = universe.drop(columns=[
        "split_source", "term_months_source", "loan_status_source", "_merge",
    ])

    snapshot = pd.to_numeric(universe["out_prncp"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid_snapshot = ~np.isfinite(snapshot) | (snapshot < 0)
    if invalid_snapshot.any():
        raise ValueError(
            f"Aborting: {int(invalid_snapshot.sum()):,} PD loans have missing, "
            "non-finite, or negative out_prncp."
        )
    universe["out_prncp_snapshot"] = snapshot

    if len(universe) != EXPECTED_LOAN_COUNT:
        raise AssertionError("The PD-authoritative universe changed during the join.")
    require_same_ids(universe, pd.Index(pd_scores["id"]), "Joined PD universe")
    return universe, pd_scores


def build_rwa_results(universe):
    current = universe["loan_status"].eq("Current")
    historical = ~current
    current_count = int(current.sum())
    if current_count != 1_789:
        raise ValueError(
            f"Expected 1,789 Current loans in this portfolio; found {current_count:,}."
        )

    total_current_exposure = float(
        universe.loc[current, "out_prncp_snapshot"].sum()
    )
    exposure_difference = abs(total_current_exposure - EXPECTED_CURRENT_EXPOSURE)
    if exposure_difference >= EXPOSURE_RECONCILIATION_TOLERANCE:
        raise ValueError(
            "Current portfolio exposure reconciliation failed: "
            f"actual={total_current_exposure:.2f}; "
            f"expected={EXPECTED_CURRENT_EXPOSURE:.2f}; "
            f"absolute_difference={exposure_difference:.6f}."
        )

    zero_balance_current = current & universe["out_prncp_snapshot"].eq(0)
    output = pd.DataFrame(index=universe.index)
    output["id"] = universe["id"]
    output["split"] = universe["split"]
    output["loan_status"] = universe["loan_status"]
    output["term_months"] = universe["term_months"]
    output["purpose"] = universe["purpose"]
    output["application_type"] = universe["application_type"]
    output["out_prncp_snapshot"] = universe["out_prncp_snapshot"]
    output["exposure_basis"] = np.where(
        current, CURRENT_EXPOSURE_BASIS, NONCURRENT_EXPOSURE_BASIS
    )
    output["illustrative_exposure_class"] = np.where(
        current,
        "Consumer lending proxy; regulatory exposure class unverified.",
        None,
    )
    output["risk_weight_assumption"] = np.where(
        current, BASELINE_RISK_WEIGHT, np.nan
    )
    output["collateral_recognized"] = COLLATERAL_UNAVAILABLE
    output["ccf_assumption"] = np.where(current, CURRENT_CCF_BASIS, None)
    output["rwa_illustrative"] = np.where(
        current,
        universe["out_prncp_snapshot"] * BASELINE_RISK_WEIGHT,
        np.nan,
    )
    output["included_in_current_portfolio"] = current.to_numpy(dtype=bool)
    output["exclusion_reason"] = np.where(
        universe["loan_status"].eq("Fully Paid"),
        "Fully Paid historical/closed loan; excluded from live portfolio.",
        np.where(
            universe["loan_status"].eq("Charged Off"),
            "Historical Charged Off loan; not treated as a live exposure.",
            None,
        ),
    )
    output["calculation_basis"] = CALCULATION_BASIS
    output["assumption_label"] = np.where(
        current, CURRENT_ASSUMPTION_LABEL, EXCLUDED_ASSUMPTION_LABEL
    )
    output = output[OUTPUT_COLUMNS].copy()

    sensitivity_totals = {
        weight: total_current_exposure * weight
        for weight in SENSITIVITY_RISK_WEIGHTS
    }
    return output, current, zero_balance_current, total_current_exposure, sensitivity_totals


def optional_ecl_comparison(pd_scores, current, baseline_rwa):
    if not ECL_IN.exists():
        print("  Optional ECL comparison unavailable: ecl_results.parquet not found.")
        return
    ecl = pd.read_parquet(ECL_IN)
    require_same_ids(ecl, pd.Index(pd_scores["id"]), "ECL results")
    required = {"loan_status", "split", "ecl_selected_synthetic"}
    missing = sorted(required.difference(ecl.columns))
    if missing:
        raise ValueError(f"ECL results are missing diagnostic columns: {missing}")
    compared = pd_scores[["id", "split", "loan_status"]].merge(
        ecl[["id", "split", "loan_status", "ecl_selected_synthetic"]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_pd", "_ecl"),
    )
    if (compared["split_pd"].ne(compared["split_ecl"]).any()
            or compared["loan_status_pd"].ne(compared["loan_status_ecl"]).any()):
        raise ValueError("ECL comparison split/status differs from PD scores.")
    current_ecl = compared.loc[
        compared["loan_status_pd"].eq("Current"), "ecl_selected_synthetic"
    ]
    if current_ecl.isna().any() or not np.isfinite(current_ecl.to_numpy(float)).all():
        raise ValueError("Selected ECL is missing/invalid for one or more Current loans.")
    selected_ecl_total = float(current_ecl.sum())
    ratio = selected_ecl_total / baseline_rwa if baseline_rwa else np.nan
    print("  Optional selected ECL for Current loans: "
          f"{selected_ecl_total:,.2f}")
    print(f"  Baseline illustrative RWA for same Current loans: {baseline_rwa:,.2f}")
    print(f"  ECL/RWA ratio (descriptive comparison only): {ratio:.6f}; "
          "not a reconciliation or regulatory capital measure.")


def validate_output(output, pd_scores, current, zero_balance_current):
    if output.columns.tolist() != OUTPUT_COLUMNS:
        raise AssertionError(
            f"RWA output schema/order changed: {output.columns.tolist()}"
        )
    if len(output) != EXPECTED_LOAN_COUNT:
        raise AssertionError("RWA output must contain exactly 66,688 loans.")
    if output["id"].isna().any() or output["id"].duplicated().any():
        raise AssertionError("RWA output IDs must be non-null and unique.")
    expected = pd_scores.set_index("id").loc[output["id"]]
    if (not output["split"].eq(expected["split"].to_numpy()).all()
            or not output["term_months"].eq(expected["term_months"].to_numpy()).all()
            or not output["loan_status"].eq(expected["loan_status"].to_numpy()).all()):
        raise AssertionError("RWA output does not preserve PD splits/terms/statuses.")

    included = output["included_in_current_portfolio"]
    if not included.equals(current):
        raise AssertionError("Exactly all Current loans must be included, and no others.")
    if not output.loc[current, "risk_weight_assumption"].eq(
        BASELINE_RISK_WEIGHT
    ).all():
        raise AssertionError("Current loans must carry the hypothetical 100% baseline weight.")
    if output.loc[~current, "risk_weight_assumption"].notna().any():
        raise AssertionError("Non-Current risk-weight assumptions must be null.")
    if output.loc[~current, "ccf_assumption"].notna().any():
        raise AssertionError("Non-Current CCF assumptions must be null.")
    if output.loc[current, "ccf_assumption"].isna().any():
        raise AssertionError("Current on-balance-sheet proxy must state CCF is not applicable.")
    if output.loc[~current, "rwa_illustrative"].notna().any():
        raise AssertionError("Non-Current RWA must remain null.")
    if not output.loc[current, "rwa_illustrative"].notna().all():
        raise AssertionError("Every Current loan, including zero-balance loans, needs RWA.")
    if not output.loc[zero_balance_current, "rwa_illustrative"].eq(0).all():
        raise AssertionError("Zero-balance Current loans must have zero illustrative RWA.")
    if output.loc[current, "out_prncp_snapshot"].lt(0).any():
        raise AssertionError("Current exposure proxies must be nonnegative.")
    if not output["collateral_recognized"].eq(COLLATERAL_UNAVAILABLE).all():
        raise AssertionError("Collateral status must be explicitly marked unavailable.")


def main():
    print("Loading PD-authoritative loan universe ...")
    universe, pd_scores = load_current_portfolio()
    current = universe["loan_status"].eq("Current")
    total_current_exposure = float(
        universe.loc[current, "out_prncp_snapshot"].sum()
    )
    exposure_difference = abs(total_current_exposure - EXPECTED_CURRENT_EXPOSURE)
    if exposure_difference >= EXPOSURE_RECONCILIATION_TOLERANCE:
        raise ValueError(
            "Current portfolio exposure reconciliation failed: "
            f"actual={total_current_exposure:.2f}; "
            f"expected={EXPECTED_CURRENT_EXPOSURE:.2f}; "
            f"absolute_difference={exposure_difference:.6f}."
        )

    output, current, zero_balance_current, total_current_exposure, sensitivity_totals = (
        build_rwa_results(universe)
    )
    validate_output(output, pd_scores, current, zero_balance_current)

    print("\nIllustrative RWA summary (Current portfolio only):")
    print(f"  PD-universe loans processed: {len(output):,}")
    print(f"  Current loans included: {int(current.sum()):,}; "
          f"zero-balance Current loans included: {int(zero_balance_current.sum()):,}")
    print(f"  Current recorded exposure total: {total_current_exposure:,.2f}; "
          f"reconciliation difference={exposure_difference:.6f}")
    print("  Hypothetical risk-weight sensitivity totals (not regulatory risk weights):")
    for weight, total in sensitivity_totals.items():
        print(f"    {weight:.0%}: {total:,.2f}")
    print(f"  Baseline illustrative RWA (hypothetical 100%): "
          f"{sensitivity_totals[BASELINE_RISK_WEIGHT]:,.2f}")
    optional_ecl_comparison(
        pd_scores,
        current,
        sensitivity_totals[BASELINE_RISK_WEIGHT],
    )
    print("  Caveat: The source out_prncp balance has no separately verified as-of timestamp. "
          "The pipeline's 2018-12-31 snapshot assumption does not establish the actual "
          "balance reporting date.")
    print("  Limitations: collateral/guarantee eligibility, regulatory exposure class, "
          "actual CCF, supervisory IRB approval, and applicable Basel/CRR treatment "
          "are not established by these files.")
    print("  Validation passed: exact 17-column schema, 66,688 unique PD IDs, all and only "
          "Current loans included, 36 zero-balance Current loans yield zero RWA, and "
          "historical loans have null RWA.")

    output.to_parquet(OUT, index=False, compression="zstd")
    print(f"Saved educational RWA results to {OUT}")
    print("Only rwa_results.parquet is written; source and model outputs are read-only.")


if __name__ == "__main__":
    main()