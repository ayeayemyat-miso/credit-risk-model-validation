"""Phase 4: projected contractual exposure-at-default schedules.

The schedule reports outstanding principal at the START of each contractual
month, before that month's scheduled payment. It assumes no prepayments, no
missed payments, no fees, and a fixed annual nominal interest rate. It is a
projected contractual EAD proxy, not observed exposure at historical default
and not the scheduled-balance proxy used as the LGD recovery denominator.

Run from the project root: python src/ead.py
Inputs: data/processed/loans_clean.parquet and pd_scores.parquet
Output: data/processed/ead_schedules.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOANS_IN = ROOT / "data" / "processed" / "loans_clean.parquet"
PD_IN = ROOT / "data" / "processed" / "pd_scores.parquet"
OUT = ROOT / "data" / "processed" / "ead_schedules.parquet"
VALID_TERMS = (36, 60)


def validate_and_join_universe(loans, pd_scores):
    """Validate IDs/splits and return one row per authoritative PD loan."""
    for frame, label in ((loans, "clean loan"), (pd_scores, "PD")):
        if "id" not in frame:
            raise ValueError(f"{label} data is missing required column 'id'.")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{label} loan IDs must be non-null and unique.")

    required_pd = {"split"}
    missing_pd = sorted(required_pd.difference(pd_scores.columns))
    if missing_pd:
        raise ValueError(f"PD scores are missing required columns: {missing_pd}")
    if not pd_scores["split"].isin(["train", "test"]).all():
        raise ValueError("PD split must contain only 'train' or 'test'.")

    required_loan = {"funded_amnt", "int_rate", "term_months", "split"}
    missing_loan = sorted(required_loan.difference(loans.columns))
    if missing_loan:
        raise ValueError(f"Clean loans are missing required columns: {missing_loan}")

    source = loans[["id", "funded_amnt", "int_rate", "term_months", "split"]]
    source = source.rename(columns={"split": "source_split"})
    universe = pd_scores[["id", "split"]].merge(
        source,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_source = universe["_merge"].ne("both")
    if missing_source.any():
        raise ValueError(
            f"{int(missing_source.sum()):,} PD loan IDs are missing from clean loans."
        )

    split_mismatch = universe["split"].ne(universe["source_split"])
    if split_mismatch.any():
        raise ValueError(
            f"{int(split_mismatch.sum()):,} PD/source train-test splits disagree."
        )
    universe = universe.drop(columns=["source_split", "_merge"])
    return universe


def coerce_and_validate_inputs(universe):
    """Report malformed schedule inputs and fail instead of dropping loans."""
    invalid_masks = {}
    for column in ("funded_amnt", "int_rate", "term_months"):
        raw = universe[column]
        values = pd.to_numeric(raw, errors="coerce")
        missing = raw.isna()
        nonnumeric = raw.notna() & values.isna()
        nonfinite = values.notna() & ~np.isfinite(values)

        if column == "funded_amnt":
            invalid_domain = values.notna() & np.isfinite(values) & values.le(0)
            domain_label = "<= 0"
        elif column == "int_rate":
            invalid_domain = values.notna() & np.isfinite(values) & values.lt(0)
            domain_label = "< 0% annual"
        else:
            invalid_domain = values.notna() & np.isfinite(values) & ~values.isin(VALID_TERMS)
            domain_label = f"not in {VALID_TERMS}"

        invalid_masks[column] = missing | nonnumeric | nonfinite | invalid_domain
        print(
            f"  {column}: missing={int(missing.sum()):,}; "
            f"nonnumeric={int(nonnumeric.sum()):,}; "
            f"nonfinite={int(nonfinite.sum()):,}; "
            f"invalid ({domain_label})={int(invalid_domain.sum()):,}"
        )
        universe[column] = values

    total_invalid = int(pd.DataFrame(invalid_masks).any(axis=1).sum())
    if total_invalid:
        raise ValueError(
            f"Cannot build schedules: {total_invalid:,} PD loans have missing or "
            "invalid funded_amnt, int_rate, or term_months values."
        )
    universe["term_months"] = universe["term_months"].astype(np.int16)
    return universe


def build_schedule(universe):
    """Build all loan-month balances with vectorized term-batched operations."""
    loan_ids = universe["id"].to_numpy()
    principal = universe["funded_amnt"].to_numpy(dtype=np.float64)
    monthly_rate = universe["int_rate"].to_numpy(dtype=np.float64) / 100.0 / 12.0
    terms = universe["term_months"].to_numpy(dtype=np.int16)
    splits = universe["split"].to_numpy()
    schedule_parts = []
    maximum_final_balance = 0.0

    for term in VALID_TERMS:
        loan_positions = np.flatnonzero(terms == term)
        if not len(loan_positions):
            continue

        term_principal = principal[loan_positions]
        term_rate = monthly_rate[loan_positions]
        zero_rate = term_rate == 0
        payment = np.empty(len(loan_positions), dtype=np.float64)
        payment[zero_rate] = term_principal[zero_rate] / term
        if (~zero_rate).any():
            log_growth = np.log1p(term_rate[~zero_rate])
            payment[~zero_rate] = (
                term_principal[~zero_rate] * term_rate[~zero_rate]
                / -np.expm1(-term * log_growth)
            )

        repeated_positions = np.repeat(loan_positions, term)
        month = np.tile(np.arange(1, term + 1, dtype=np.int16), len(loan_positions))
        remaining_payments = term - month.astype(np.int32) + 1
        row_rate = np.repeat(term_rate, term)
        row_payment = np.repeat(payment, term)
        row_zero_rate = row_rate == 0
        ead = np.empty(len(month), dtype=np.float64)
        ead[row_zero_rate] = (
            row_payment[row_zero_rate] * remaining_payments[row_zero_rate]
        )
        positive_rate = ~row_zero_rate
        ead[positive_rate] = (
            row_payment[positive_rate]
            * -np.expm1(
                -remaining_payments[positive_rate]
                * np.log1p(row_rate[positive_rate])
            )
            / row_rate[positive_rate]
        )

        balances_by_loan = ead.reshape(len(loan_positions), term)
        if not np.isfinite(ead).all() or (ead < 0).any():
            raise AssertionError(f"Term {term}: EAD values must be finite and nonnegative.")
        if not np.allclose(
            balances_by_loan[:, 0], term_principal, rtol=1e-12, atol=1e-8
        ):
            raise AssertionError(f"Term {term}: month 1 EAD must equal funded_amnt.")
        if (np.diff(balances_by_loan, axis=1) > 1e-8).any():
            raise AssertionError(f"Term {term}: monthly EAD must be nonincreasing.")

        final_start = balances_by_loan[:, -1]
        final_balance = final_start * (1.0 + term_rate) - payment
        maximum_final_balance = max(
            maximum_final_balance, float(np.max(np.abs(final_balance)))
        )
        if not np.allclose(final_balance, 0.0, rtol=1e-10, atol=1e-7):
            raise AssertionError(
                f"Term {term}: balance after final scheduled payment is not zero."
            )

        schedule_parts.append(pd.DataFrame({
            "id": loan_ids[repeated_positions],
            "month": month,
            "ead": ead,
            "term_months": np.full(len(month), term, dtype=np.int16),
            "split": splits[repeated_positions],
        }))

    if not schedule_parts:
        raise ValueError("No valid PD loans were available to schedule.")
    schedule = pd.concat(schedule_parts, ignore_index=True)
    schedule = schedule.sort_values(["id", "month"], kind="stable").reset_index(drop=True)

    if schedule.duplicated(["id", "month"]).any():
        raise AssertionError("Duplicate (id, month) schedule rows were generated.")
    if not np.isfinite(schedule["ead"].to_numpy(dtype=np.float64)).all():
        raise AssertionError("Schedule contains non-finite EAD values.")
    if schedule["ead"].lt(0).any():
        raise AssertionError("Schedule contains negative EAD values.")
    if not schedule["month"].between(1, schedule["term_months"]).all():
        raise AssertionError("Schedule month is outside its contractual term.")

    return schedule, maximum_final_balance


def main():
    print("Loading PD universe and clean loan data ...")
    loans = pd.read_parquet(LOANS_IN)
    pd_scores = pd.read_parquet(PD_IN)
    print(f"  Clean loans: {len(loans):,}")
    print(f"  Authoritative PD loans: {len(pd_scores):,}")

    universe = validate_and_join_universe(loans, pd_scores)
    print(f"  One-to-one PD universe joined: {len(universe):,}")
    print("Validating schedule inputs:")
    universe = coerce_and_validate_inputs(universe)

    schedule, maximum_final_balance = build_schedule(universe)
    loan_counts = schedule.groupby("id", sort=False).size().reindex(universe["id"])
    expected_counts = universe["term_months"].to_numpy(dtype=np.int64)
    if not np.array_equal(loan_counts.to_numpy(dtype=np.int64), expected_counts):
        raise AssertionError("Each PD loan must have one row for every contractual month.")
    if schedule["id"].nunique() != len(pd_scores):
        raise AssertionError("Schedule must preserve every PD loan ID.")

    print("\nEAD schedule summary:")
    print(f"  Loans: {schedule['id'].nunique():,}; rows: {len(schedule):,}")
    print(f"  Loans by split: {universe['split'].value_counts().to_dict()}")
    print("  Schedule rows by term:")
    print(schedule.groupby("term_months").size().to_string())
    print("  EAD summary (all loan-month rows):")
    print(schedule["ead"].describe().round(2).to_string())
    print("  EAD summary by term:")
    print(schedule.groupby("term_months")["ead"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    ).round(2).to_string())
    print(f"  Maximum absolute balance after final scheduled payment: "
          f"{maximum_final_balance:.3e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(OUT, index=False, compression="zstd")
    print(f"Saved compressed schedule to {OUT}")


if __name__ == "__main__":
    main()