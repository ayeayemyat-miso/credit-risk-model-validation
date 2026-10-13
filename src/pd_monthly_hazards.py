"""Generate loan-level monthly conditional PD hazards for downstream ECL work.

The 36- and 60-month fitted models are not persisted, so this script retrains
them by calling the existing PD preparation and fitting functions unchanged.
It writes only data/processed/pd_monthly_hazards.parquet; existing PD outputs
are read for universe and reconciliation checks, never modified.

These are origination-based conditional hazards, not reporting-date IFRS 9 PDs.
Training-loan predictions are in-sample. Test predictions are out-of-sample
within the existing deterministic 15% sample.

Run from project root: python src/pd_monthly_hazards.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import pd as pd_model
else:
    import pd as pd_model

ROOT = Path(__file__).resolve().parent.parent
PD_SCORES_IN = ROOT / "data" / "processed" / "pd_scores.parquet"
OUT = ROOT / "data" / "processed" / "pd_monthly_hazards.parquet"
EXPECTED_LOAN_COUNT = 66_688
RECONCILIATION_ATOL = 1e-9


def validate_recreated_universe(meta, saved_scores):
    """Require the exact scored ID, split, and term universe from pd.py."""
    if len(saved_scores) != EXPECTED_LOAN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOAN_COUNT:,} saved PD scores; found {len(saved_scores):,}."
        )
    if saved_scores["id"].isna().any() or saved_scores["id"].duplicated().any():
        raise ValueError("Saved PD score IDs must be non-null and unique.")
    if meta["id"].isna().any() or meta["id"].duplicated().any():
        raise ValueError("Recreated PD metadata IDs must be non-null and unique.")

    saved = saved_scores[["id", "split", "term_months"]]
    recreated = meta[["id", "split", "term_months"]]
    check = saved.merge(
        recreated,
        on="id",
        how="outer",
        validate="one_to_one",
        suffixes=("_saved", "_recreated"),
        indicator=True,
    )
    missing_or_added = check["_merge"].ne("both")
    if missing_or_added.any():
        raise ValueError(
            f"Recreated PD universe differs by {int(missing_or_added.sum()):,} IDs."
        )
    split_mismatch = check["split_saved"].ne(check["split_recreated"])
    term_mismatch = check["term_months_saved"].ne(check["term_months_recreated"])
    if split_mismatch.any() or term_mismatch.any():
        raise ValueError(
            "Recreated PD metadata does not preserve the saved split/term: "
            f"split_mismatches={int(split_mismatch.sum()):,}; "
            f"term_mismatches={int(term_mismatch.sum()):,}."
        )


def marginal_pds_from_hazards(hazard_matrix):
    """Derive marginal default probabilities using the PD pipeline's clipping."""
    safe_hazard = np.clip(
        hazard_matrix,
        0.0,
        np.nextafter(1.0, 0.0),
    )
    survival_before_month = np.concatenate(
        [
            np.ones((len(safe_hazard), 1), dtype=np.float64),
            np.cumprod(1.0 - safe_hazard, axis=1)[:, :-1],
        ],
        axis=1,
    )
    marginal_pd = safe_hazard * survival_before_month
    return marginal_pd


def generate_hazards(df, meta, numeric_features, categorical_features, models,
                     saved_scores):
    """Score every eligible loan over its full contractual term in chunks."""
    feature_frame = df[["id"] + numeric_features + categorical_features]
    score_lookup = saved_scores.set_index("id")
    hazard_parts = []
    reconciliation = {
        "lifetime_abs_errors": [],
        "pd12_abs_errors": [],
        "lifetime_failures": 0,
        "pd12_failures": 0,
        "loans": 0,
    }

    for term in pd_model.TERMS:
        selected = meta.loc[
            meta["term_months"].eq(term), ["id", "split", "term_months"]
        ]
        model, transformers = models[term]
        print(f"Scoring {len(selected):,} eligible loans over {term} months ...")

        for start in range(0, len(selected), pd_model.CHUNK_SIZE):
            batch = selected.iloc[start:start + pd_model.CHUNK_SIZE].merge(
                feature_frame,
                on="id",
                how="left",
                validate="one_to_one",
            )
            durations = batch["term_months"].astype(int)
            panel = pd_model.expand_panel(
                batch[["id"] + numeric_features + categorical_features],
                durations,
            )
            design = pd_model.transform(
                panel, numeric_features, categorical_features, transformers
            )
            hazard = model.predict_proba(design)[:, 1].astype(np.float64)
            if not np.isfinite(hazard).all() or (hazard < 0).any() or (hazard > 1).any():
                raise AssertionError("Predicted conditional hazards must be finite in [0, 1].")
            if len(hazard) != len(batch) * term:
                raise AssertionError(f"Term {term}: unexpected number of loan-month hazards.")

            hazard_matrix = hazard.reshape(len(batch), term)
            marginal_pd = marginal_pds_from_hazards(hazard_matrix)
            derived_lifetime = marginal_pd.sum(axis=1)
            derived_pd12 = marginal_pd[:, :12].sum(axis=1)
            saved_batch = score_lookup.loc[batch["id"].to_numpy()]
            lifetime_error = np.abs(
                derived_lifetime - saved_batch["pd_lifetime"].to_numpy(dtype=np.float64)
            )
            pd12_error = np.abs(
                derived_pd12 - saved_batch["pd_12m"].to_numpy(dtype=np.float64)
            )
            reconciliation["lifetime_abs_errors"].extend(lifetime_error.tolist())
            reconciliation["pd12_abs_errors"].extend(pd12_error.tolist())
            reconciliation["lifetime_failures"] += int(
                np.count_nonzero(lifetime_error > RECONCILIATION_ATOL)
            )
            reconciliation["pd12_failures"] += int(
                np.count_nonzero(pd12_error > RECONCILIATION_ATOL)
            )
            reconciliation["loans"] += len(batch)

            hazard_parts.append(pd.DataFrame({
                "id": panel["id"].to_numpy(),
                "split": np.repeat(batch["split"].to_numpy(), term),
                "term_months": np.full(len(panel), term, dtype=np.int16),
                "month": panel["month"].to_numpy(dtype=np.int16),
                "hazard": hazard,
            }))
            del panel, design, hazard, hazard_matrix, marginal_pd

    if not hazard_parts:
        raise ValueError("No eligible loans were scored for monthly hazards.")
    hazards = pd.concat(hazard_parts, ignore_index=True)
    return hazards, reconciliation


def validate_hazard_output(hazards, meta):
    required_columns = ["id", "split", "term_months", "month", "hazard"]
    if hazards.columns.tolist() != required_columns:
        raise AssertionError(f"Unexpected output schema: {hazards.columns.tolist()}")
    if hazards.duplicated(["id", "month"]).any():
        raise AssertionError("Monthly hazard output contains duplicate (id, month) keys.")
    values = hazards["hazard"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("All hazard values must be finite and within [0, 1].")
    if not hazards["month"].between(1, hazards["term_months"]).all():
        raise AssertionError("Hazard months must fall within their contractual terms.")

    loan_counts = hazards.groupby("id", sort=False).size().reindex(meta["id"])
    expected_counts = meta["term_months"].to_numpy(dtype=np.int64)
    if loan_counts.isna().any() or not np.array_equal(
        loan_counts.to_numpy(dtype=np.int64), expected_counts
    ):
        raise AssertionError("Each eligible loan must have exactly term_months rows.")

    loan_metadata = hazards.groupby("id", sort=False).agg(
        split_count=("split", "nunique"),
        term_count=("term_months", "nunique"),
        split=("split", "first"),
        term_months=("term_months", "first"),
    )
    expected = meta.set_index("id").loc[loan_metadata.index]
    if (loan_metadata["split_count"].ne(1).any()
            or loan_metadata["term_count"].ne(1).any()
            or not loan_metadata["split"].eq(expected["split"]).all()
            or not loan_metadata["term_months"].eq(expected["term_months"]).all()):
        raise AssertionError("Hazard output split/term does not match PD metadata.")


def main():
    saved_scores = pd.read_parquet(PD_SCORES_IN)
    required_scores = {"id", "split", "term_months", "pd_12m", "pd_lifetime"}
    missing_scores = sorted(required_scores.difference(saved_scores.columns))
    if missing_scores:
        raise ValueError(f"Saved PD scores are missing required columns: {missing_scores}")

    print("Reusing src/pd.py preparation, training split, features, and model specification.")
    print("No fitted PD model artifacts were found; both term models will be retrained.")
    df, numeric_features, categorical_features = pd_model.load_and_prepare()
    meta = pd_model.observation_metadata(df)
    validate_recreated_universe(meta, saved_scores)

    # This is the unchanged PD training routine: separate models per term,
    # train-only fitting, the original 15% sample, features, and hyperparameters.
    models = pd_model.fit_models(
        df, meta, numeric_features, categorical_features
    )
    hazards, reconciliation = generate_hazards(
        df,
        meta,
        numeric_features,
        categorical_features,
        models,
        saved_scores,
    )
    validate_hazard_output(hazards, meta)

    lifetime_errors = np.asarray(reconciliation["lifetime_abs_errors"])
    pd12_errors = np.asarray(reconciliation["pd12_abs_errors"])
    print("\nMonthly hazard validation:")
    print(f"  Loans scored: {reconciliation['loans']:,}; "
          f"loan-month rows: {len(hazards):,}")
    print(f"  Rows by term: {hazards.groupby('term_months').size().to_dict()}")
    print(f"  Rows by split: {hazards.drop_duplicates('id').split.value_counts().to_dict()}")
    print(f"  Unique (id, month) keys: {not hazards.duplicated(['id', 'month']).any()}")
    print(f"  Hazard range: [{hazards['hazard'].min():.8g}, "
          f"{hazards['hazard'].max():.8g}]; finite={np.isfinite(hazards['hazard']).all()}")
    print(f"  Lifetime PD reconciliation: failures > {RECONCILIATION_ATOL:g} = "
          f"{reconciliation['lifetime_failures']:,}; "
          f"max abs error={lifetime_errors.max():.3e}")
    print(f"  12-month PD reconciliation: failures > {RECONCILIATION_ATOL:g} = "
          f"{reconciliation['pd12_failures']:,}; "
          f"max abs error={pd12_errors.max():.3e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hazards.to_parquet(OUT, index=False, compression="zstd")
    print(f"Saved loan-level monthly hazards to {OUT}")
    print("Limitations: train hazards are in-sample; test hazards are out-of-sample "
          "within the existing stratified 15% sample. Results are origination-based, "
          "not reporting-date ECL PDs.")


if __name__ == "__main__":
    main()