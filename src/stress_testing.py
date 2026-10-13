"""Educational ECL sensitivities for Current synthetic Stage 1/2 loans.

This is not reporting-date IFRS 9 or regulatory stress testing. It stresses
origination-based PD and the constant historical training-mean LGD against the
existing contractual EAD schedule. Synthetic stage membership stays fixed.

Run from the project root: python src/stress_testing.py
Only the requested stress reports and figure are written.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
STAGING_IN = PROCESSED / "staging.parquet"
HAZARDS_IN = PROCESSED / "pd_monthly_hazards.parquet"
EAD_IN = PROCESSED / "ead_schedules.parquet"
LGD_IN = PROCESSED / "lgd_scores.parquet"
ECL_IN = PROCESSED / "ecl_results.parquet"
LOANS_IN = PROCESSED / "loans_clean.parquet"
RWA_IN = PROCESSED / "rwa_results.parquet"
SUMMARY_OUT = REPORTS / "stress_testing_summary.csv"
BY_STAGE_OUT = REPORTS / "stress_testing_by_stage.csv"
FIGURE_OUT = FIGURES / "stress_testing_ecl.png"

EXPECTED_CURRENT_LOANS = 1_789
EXPECTED_STAGE_COUNTS = {"Synthetic Stage 1": 1_275, "Synthetic Stage 2": 514}
EXPECTED_MONTH_ROWS = 107_340
EXPECTED_BASELINE_ECL = 1_285_357.2530451622
BASELINE_LOAN_TOL = 1e-6
BASELINE_TOTAL_TOL = 0.01
PROBABILITY_TOL = 1e-12
ECL_TOL = 1e-8
TERM = 60
STAGES = ("Synthetic Stage 1", "Synthetic Stage 2")
SCENARIOS = (
    {"scenario": "Baseline", "pd_multiplier": 1.00, "lgd_uplift_pp": 0.0},
    {"scenario": "PD stress", "pd_multiplier": 1.25, "lgd_uplift_pp": 0.0},
    {"scenario": "LGD stress", "pd_multiplier": 1.00, "lgd_uplift_pp": 10.0},
    {"scenario": "Combined adverse", "pd_multiplier": 1.50, "lgd_uplift_pp": 10.0},
)
CALCULATION_BASIS = (
    "Educational origination-based ECL sensitivity; proportional scaling of monthly "
    "marginal PDs to capped cumulative-PD targets; fixed synthetic stages; "
    "contractual start-of-month EAD and int_rate discount proxy. Not calibrated, "
    "reporting-date, IFRS 9, or regulatory stress testing."
)
BASIS = (
    "Educational origination-based ECL sensitivity; survival-consistent capped "
    "cumulative-PD multiplier; fixed synthetic stage; contractual start-month "
    "EAD and contractual-rate discount proxy. Not calibrated macroeconomic, "
    "reporting-date, IFRS 9, or regulatory stress testing."
)


def require_unique_ids(frame, label):
    if "id" not in frame or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{label} must have non-null unique IDs.")


def require_same_ids(frame, ids, label):
    require_unique_ids(frame, label)
    actual = pd.Index(frame["id"])
    missing, extra = ids.difference(actual), actual.difference(ids)
    if len(actual) != len(ids) or len(missing) or len(extra):
        raise ValueError(
            f"{label} ID mismatch: expected {len(ids)}, actual {len(actual)}, "
            f"missing {len(missing)}, extra {len(extra)}."
        )


def load_current():
    staging = pd.read_parquet(STAGING_IN, columns=[
        "id", "split", "term_months", "loan_status",
        "observed_classification", "synthetic_stage",
    ])
    require_unique_ids(staging, "Staging output")
    current = staging.loc[staging.loan_status.eq("Current")].copy()
    require_unique_ids(current, "Current staging population")
    counts = current.synthetic_stage.value_counts().to_dict()
    if len(current) != EXPECTED_CURRENT_LOANS or counts != EXPECTED_STAGE_COUNTS:
        raise ValueError(f"Unexpected Current count/stage mix: n={len(current)}, stages={counts}.")
    if current.term_months.ne(TERM).any() or not current.observed_classification.eq("Unassessed").all():
        raise ValueError("Current stage metadata has unexpected term/classification.")
    if current.synthetic_stage.isna().any() or not set(current.synthetic_stage).issubset(STAGES):
        raise ValueError("Each Current loan must have a Synthetic Stage 1/2 label.")
    current = current.sort_values("id", kind="stable").reset_index(drop=True)
    current_ids = pd.Index(current.id)

    loans = pd.read_parquet(
        LOANS_IN,
        columns=["id", "split", "term_months", "loan_status", "int_rate"],
    )
    require_unique_ids(loans, "Clean loans")
    loans = loans.loc[loans.id.isin(current_ids)].rename(columns={
        "split": "split_loan", "term_months": "term_loan", "loan_status": "status_loan",
    })
    require_same_ids(loans, current_ids, "Current clean loans")
    current = current.merge(loans, on="id", how="left", validate="one_to_one")
    if (not current.split.eq(current.split_loan).all()
            or not current.term_months.eq(current.term_loan).all()
            or not current.loan_status.eq(current.status_loan).all()):
        raise ValueError("Staging and clean-loan Current split/term/status disagree.")
    current = current.drop(columns=["split_loan", "term_loan", "status_loan"])

    lgd = pd.read_parquet(LGD_IN, columns=["id", "split", "loan_status", "lgd_predicted"])
    lgd = lgd.loc[lgd.id.isin(current_ids)].rename(columns={
        "split": "split_lgd", "loan_status": "status_lgd",
    })
    require_same_ids(lgd, current_ids, "Current LGD records")
    current = current.merge(lgd, on="id", how="left", validate="one_to_one")
    if (not current.split.eq(current.split_lgd).all()
            or not current.loan_status.eq(current.status_lgd).all()):
        raise ValueError("Staging and LGD Current split/status disagree.")
    current = current.drop(columns=["split_lgd", "status_lgd"])
    current["lgd_baseline"] = pd.to_numeric(current.lgd_predicted, errors="coerce")
    lgd_values = current.lgd_baseline.to_numpy(dtype=np.float64)
    if not np.isfinite(lgd_values).all() or (lgd_values < 0).any() or (lgd_values > 1).any():
        raise ValueError("Current baseline LGD must be finite within [0, 1].")

    current.int_rate = pd.to_numeric(current.int_rate, errors="coerce")
    rates = current.int_rate.to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 100).any():
        raise ValueError("Current int_rate must be a finite annual percentage in [0, 100].")
    return current


def load_current_monthly(current):
    ids = pd.Index(current.id)
    hazards = pd.read_parquet(
        HAZARDS_IN,
        columns=["id", "split", "term_months", "month", "hazard"],
        filters=[("term_months", "==", TERM)],
    )
    hazards = hazards.loc[hazards.id.isin(ids)].copy()
    ead = pd.read_parquet(
        EAD_IN,
        columns=["id", "split", "term_months", "month", "ead"],
        filters=[("term_months", "==", TERM)],
    )
    ead = ead.loc[ead.id.isin(ids)].copy()
    for frame, label in ((hazards, "hazards"), (ead, "EAD")):
        if frame.id.isna().any() or frame.duplicated(["id", "month"]).any():
            raise ValueError(f"Current {label} have missing IDs or duplicate (id, month) keys.")
        if (not frame.term_months.eq(TERM).all()
                or not frame.month.between(1, TERM).all()):
            raise ValueError(f"Current {label} have invalid term/month values.")
        counts = frame.groupby("id", sort=False).size()
        if len(counts) != EXPECTED_CURRENT_LOANS or not counts.eq(TERM).all():
            raise ValueError(f"Current {label} do not have exactly 60 monthly rows per loan.")
        coverage = frame.groupby("id").month.agg(["min", "max", "nunique"])
        if (coverage["min"].ne(1).any() or coverage["max"].ne(TERM).any()
                or coverage["nunique"].ne(TERM).any()):
            raise ValueError(f"Current {label} do not cover months 1..60.")
        require_same_ids(counts.rename("rows").reset_index(), ids, f"Current {label}")

    key_check = hazards[["id", "month"]].merge(
        ead[["id", "month"]], on=["id", "month"], how="outer",
        validate="one_to_one", indicator=True,
    )
    if not key_check._merge.eq("both").all():
        raise ValueError("Current hazard and EAD monthly keys differ.")
    monthly = hazards.merge(
        ead.rename(columns={"split": "split_ead", "term_months": "term_ead"}),
        on=["id", "month"], how="left", validate="one_to_one", indicator=True,
    )
    if (not monthly._merge.eq("both").all()
            or not monthly.split.eq(monthly.split_ead).all()
            or not monthly.term_months.eq(monthly.term_ead).all()):
        raise ValueError("Current hazard/EAD values do not align by key/split/term.")
    monthly = monthly.drop(columns=["_merge", "split_ead", "term_ead"])
    monthly = monthly.sort_values(["id", "month"], kind="stable").reset_index(drop=True)
    if not np.array_equal(monthly.id.drop_duplicates().to_numpy(), current.id.to_numpy()):
        raise ValueError("Sorted monthly IDs differ from Current staging IDs.")

    hazard_matrix = pd.to_numeric(monthly.hazard, errors="coerce").to_numpy(float).reshape(
        EXPECTED_CURRENT_LOANS, TERM
    )
    ead_matrix = pd.to_numeric(monthly.ead, errors="coerce").to_numpy(float).reshape(
        EXPECTED_CURRENT_LOANS, TERM
    )
    if (not np.isfinite(hazard_matrix).all() or (hazard_matrix < 0).any()
            or (hazard_matrix > 1).any()):
        raise ValueError("Current hazards must be finite in [0, 1].")
    if not np.isfinite(ead_matrix).all() or (ead_matrix < 0).any():
        raise ValueError("Current EAD must be finite and nonnegative.")
    return monthly, hazard_matrix, ead_matrix


def marginal_from_hazard(hazards):
    survival_before = np.concatenate(
        [np.ones((hazards.shape[0], 1)), np.cumprod(1 - hazards, axis=1)[:, :-1]],
        axis=1,
    )
    marginal = hazards * survival_before
    survival_after = survival_before * (1 - hazards)
    return marginal, survival_before, survival_after


def stress_marginals(baseline_marginal, multiplier):
    baseline_cumulative = baseline_marginal.sum(axis=1)
    target = np.minimum(multiplier * baseline_cumulative, 1.0)
    scale = np.divide(
        target, baseline_cumulative,
        out=np.zeros_like(target),
        where=baseline_cumulative > 0,
    )
    stressed = baseline_marginal * scale[:, None]
    if (not np.isfinite(stressed).all() or (stressed < -PROBABILITY_TOL).any()
            or (stressed > 1 + PROBABILITY_TOL).any()):
        raise ValueError("Stressed marginal PDs are invalid.")
    if not np.allclose(stressed.sum(axis=1), target, rtol=0, atol=PROBABILITY_TOL):
        raise ValueError("Stressed marginal PDs do not sum to target cumulative PD.")

    prior_cumulative = np.cumsum(stressed, axis=1) - stressed
    remaining_survival = 1 - prior_cumulative
    if (remaining_survival < -PROBABILITY_TOL).any():
        raise ValueError("Stressed cumulative marginal PD exceeds one before month end.")
    exhausted = remaining_survival <= PROBABILITY_TOL
    if (stressed[exhausted] > PROBABILITY_TOL).any():
        raise ValueError("Positive stressed marginal PD remains after survival is exhausted.")
    reconstructed_hazard = np.divide(
        stressed, remaining_survival,
        out=np.zeros_like(stressed),
        where=remaining_survival > PROBABILITY_TOL,
    )
    if (not np.isfinite(reconstructed_hazard).all()
            or (reconstructed_hazard < -PROBABILITY_TOL).any()
            or (reconstructed_hazard > 1 + PROBABILITY_TOL).any()):
        raise ValueError("Reconstructed stressed hazards are outside [0, 1].")
    reconstructed_hazard = np.clip(reconstructed_hazard, 0, 1)
    rebuilt, _, _ = marginal_from_hazard(reconstructed_hazard)
    if not np.allclose(rebuilt, stressed, rtol=0, atol=PROBABILITY_TOL):
        raise ValueError("Reconstructed hazards fail to reproduce stressed marginals.")
    return stressed, target


def calculate_scenarios(current, hazards, ead):
    marginal, survival_before, survival_after = marginal_from_hazard(hazards)
    for survival in (survival_before, survival_after):
        if (not np.isfinite(survival).all() or (survival < -PROBABILITY_TOL).any()
                or (survival > 1 + PROBABILITY_TOL).any()):
            raise ValueError("Baseline survival probabilities are invalid.")
    cumulative = marginal.sum(axis=1)
    if (not np.isfinite(marginal).all() or (marginal < -PROBABILITY_TOL).any()
            or (cumulative < 0).any() or (cumulative > 1 + PROBABILITY_TOL).any()):
        raise ValueError("Baseline marginal/cumulative PDs are invalid.")

    lgd = current.lgd_baseline.to_numpy(dtype=np.float64)
    rate = current.int_rate.to_numpy(dtype=np.float64)
    months = np.arange(1, TERM + 1, dtype=np.float64)
    discount = (1 + rate[:, None] / 1200) ** (-months[None, :])
    if (not np.isfinite(lgd).all() or (lgd < 0).any() or (lgd > 1).any()
            or not np.isfinite(discount).all() or (discount <= 0).any()
            or (discount > 1).any()):
        raise ValueError("LGD or contractual-rate discount factors are invalid.")
    stage = current.synthetic_stage.to_numpy()
    stage1, stage2 = stage == STAGES[0], stage == STAGES[1]
    if not (stage1 | stage2).all():
        raise ValueError("Unexpected synthetic stage in Current population.")

    rows = []
    selected = {}
    for spec in SCENARIOS:
        stressed_q, _ = stress_marginals(marginal, spec["pd_multiplier"])
        stressed_lgd = np.minimum(lgd + spec["lgd_uplift_pp"] / 100, 1)
        monthly_loss = stressed_q * stressed_lgd[:, None] * ead * discount
        if not np.isfinite(monthly_loss).all() or (monthly_loss < 0).any():
            raise ValueError(f"{spec['scenario']} monthly ECL is invalid.")
        ecl12 = monthly_loss[:, :12].sum(axis=1)
        lifetime = monthly_loss.sum(axis=1)
        if (not np.isfinite(ecl12).all() or not np.isfinite(lifetime).all()
                or (ecl12 < 0).any() or (lifetime < 0).any()
                or (ecl12 > lifetime + ECL_TOL).any()):
            raise ValueError(f"{spec['scenario']} aggregated ECL is invalid.")
        selected_ecl = np.where(stage1, ecl12, lifetime)
        if not np.isfinite(selected_ecl).all() or (selected_ecl < 0).any():
            raise ValueError(f"{spec['scenario']} selected ECL is invalid.")
        selected[spec["scenario"]] = selected_ecl
        rows.append({
            "scenario": spec["scenario"],
            "pd_multiplier": spec["pd_multiplier"],
            "lgd_uplift_pp": spec["lgd_uplift_pp"],
            "loan_count": len(current),
            "ecl_total": float(selected_ecl.sum()),
            "stage1_ecl_total": float(ecl12[stage1].sum()),
            "stage2_ecl_total": float(lifetime[stage2].sum()),
            "calculation_basis": CALCULATION_BASIS,
        })
    baseline_selected = selected["Baseline"]
    for scenario in ("PD stress", "LGD stress", "Combined adverse"):
        if (selected[scenario] + ECL_TOL < baseline_selected).any():
            raise ValueError(f"{scenario} selected ECL is below baseline for a Current loan.")
    return rows, selected


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    current = load_current()
    monthly, hazards, ead = load_current_monthly(current)
    if len(monthly) != EXPECTED_MONTH_ROWS:
        raise AssertionError(f"Expected {EXPECTED_MONTH_ROWS:,} Current loan-months; found {len(monthly):,}.")

    print("Current portfolio inputs validated:")
    print(f"  loans={len(current):,}; stages={current.synthetic_stage.value_counts().to_dict()}; "
          f"monthly rows={len(monthly):,}; unique/aligned monthly keys=True")

    saved_ecl = pd.read_parquet(ECL_IN, columns=[
        "id", "split", "term_months", "loan_status", "synthetic_stage",
        "lgd_baseline", "contractual_int_rate_pct", "ecl_selected_synthetic",
    ])
    saved_ecl = saved_ecl.loc[saved_ecl.id.isin(current.id)].copy()
    require_same_ids(saved_ecl, pd.Index(current.id), "Current ECL reference")
    saved_ecl = saved_ecl.set_index("id").loc[current.id]
    for column, expected in (
        ("split", current.split.to_numpy()),
        ("term_months", current.term_months.to_numpy()),
        ("loan_status", current.loan_status.to_numpy()),
        ("synthetic_stage", current.synthetic_stage.to_numpy()),
    ):
        if not np.array_equal(saved_ecl[column].to_numpy(), expected):
            raise ValueError(f"Current ECL reference {column} differs from current input.")
    for column, expected in (
        ("lgd_baseline", current.lgd_baseline.to_numpy(dtype=np.float64)),
        ("contractual_int_rate_pct", current.int_rate.to_numpy(dtype=np.float64)),
    ):
        if not np.allclose(saved_ecl[column].to_numpy(dtype=np.float64), expected,
                           rtol=0, atol=1e-12):
            raise ValueError(f"ECL reference {column} differs from source inputs.")

    baseline_q, _, _ = marginal_from_hazard(hazards)
    rate = current.int_rate.to_numpy(dtype=np.float64)
    lgd = current.lgd_baseline.to_numpy(dtype=np.float64)
    months = np.arange(1, TERM + 1, dtype=np.float64)
    discount = (1 + rate[:, None] / 1200) ** (-months[None, :])
    baseline_loss = baseline_q * lgd[:, None] * ead * discount
    baseline12 = baseline_loss[:, :12].sum(axis=1)
    baseline_lifetime = baseline_loss.sum(axis=1)
    stage1 = current.synthetic_stage.eq(STAGES[0]).to_numpy()
    baseline_selected = np.where(stage1, baseline12, baseline_lifetime)
    saved_selected = saved_ecl.ecl_selected_synthetic.to_numpy(dtype=np.float64)
    per_loan_error = np.abs(baseline_selected - saved_selected)
    aggregate_error = abs(float(baseline_selected.sum() - saved_selected.sum()))
    max_loan_error = float(per_loan_error.max())
    print("Baseline selected ECL reconstruction:")
    print(f"  reconstructed={baseline_selected.sum():.10f}; saved={saved_selected.sum():.10f}; "
          f"max per-loan difference={max_loan_error:.3e}; aggregate difference={aggregate_error:.3e}")
    if max_loan_error >= BASELINE_LOAN_TOL or aggregate_error >= BASELINE_TOTAL_TOL:
        raise AssertionError("Baseline ECL reconstruction failed; stress scenarios aborted.")
    if abs(float(saved_selected.sum()) - EXPECTED_BASELINE_ECL) >= BASELINE_TOTAL_TOL:
        raise AssertionError("Saved baseline ECL differs from the expected Current total.")

    scenario_rows, selected_by_scenario = calculate_scenarios(current, hazards, ead)
    summary = pd.DataFrame(scenario_rows)
    baseline_total = float(summary.loc[summary.scenario.eq("Baseline"), "ecl_total"].iloc[0])
    summary["change_vs_baseline"] = summary.ecl_total - baseline_total
    summary["pct_change_vs_baseline"] = 100 * summary.change_vs_baseline / baseline_total
    order = [scenario["scenario"] for scenario in SCENARIOS]
    summary["scenario"] = pd.Categorical(summary.scenario, categories=order, ordered=True)
    summary = summary.sort_values("scenario").reset_index(drop=True)
    summary["scenario"] = summary.scenario.astype(str)
    summary = summary[[
        "scenario", "pd_multiplier", "lgd_uplift_pp", "loan_count", "ecl_total",
        "change_vs_baseline", "pct_change_vs_baseline", "calculation_basis",
    ]]

    baseline_stage_totals = {
        stage: float(selected_by_scenario["Baseline"][current.synthetic_stage.eq(stage).to_numpy()].sum())
        for stage in STAGES
    }
    stage_rows = []
    for scenario in order:
        for stage in STAGES:
            mask = current.synthetic_stage.eq(stage).to_numpy()
            total = float(selected_by_scenario[scenario][mask].sum())
            reference = baseline_stage_totals[stage]
            change = total - reference
            stage_rows.append({
                "scenario": scenario,
                "synthetic_stage": stage,
                "loan_count": int(mask.sum()),
                "ecl_total": total,
                "change_vs_baseline": change,
                "pct_change_vs_baseline": 100 * change / reference,
            })
    by_stage = pd.DataFrame(stage_rows)
    for scenario in order:
        selected_total = float(summary.loc[summary.scenario.eq(scenario), "ecl_total"].iloc[0])
        stage_total = float(by_stage.loc[by_stage.scenario.eq(scenario), "ecl_total"].sum())
        if abs(selected_total - stage_total) > ECL_TOL:
            raise AssertionError(f"{scenario} stage contributions do not sum to portfolio ECL.")
        if by_stage.loc[by_stage.scenario.eq(scenario), "loan_count"].sum() != EXPECTED_CURRENT_LOANS:
            raise AssertionError(f"{scenario} changed fixed stage membership/counts.")

    rwa = pd.read_parquet(RWA_IN, columns=[
        "id", "loan_status", "included_in_current_portfolio", "rwa_illustrative",
    ])
    rwa = rwa.loc[rwa.id.isin(current.id)].copy()
    require_same_ids(rwa, pd.Index(current.id), "Current RWA reference")
    if (not rwa.included_in_current_portfolio.all()
            or not rwa.loan_status.eq("Current").all() or rwa.rwa_illustrative.isna().any()):
        raise ValueError("Existing RWA output does not match the Current population.")
    rwa_total = float(rwa.rwa_illustrative.sum())

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False, float_format="%.6f")
    by_stage.to_csv(BY_STAGE_OUT, index=False, float_format="%.6f")
    chart_data = by_stage.pivot(index="scenario", columns="synthetic_stage", values="ecl_total").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(order), dtype=np.float64)
    colors = {STAGES[0]: "#3976a8", STAGES[1]: "#e38b42"}
    for stage in STAGES:
        values = chart_data[stage].to_numpy(dtype=np.float64)
        ax.bar(order, values, bottom=bottom, label=stage, color=colors[stage])
        bottom += values
    ax.set_ylabel("Selected illustrative ECL (source-currency units)")
    ax.set_title("Current-portfolio educational ECL stress sensitivities")
    ax.legend(title="Fixed synthetic stage")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=160)
    plt.close(fig)

    print("\nScenario assumptions and selected ECL (source-currency units):")
    print(f"  Current loans={len(current):,}; Stage 1={int(stage1.sum()):,}; "
          f"Stage 2={int((~stage1).sum()):,}; monthly rows={len(monthly):,}")
    print(f"  Baseline reconciliation passed: max loan error={max_loan_error:.3e}; "
          f"aggregate error={aggregate_error:.3e}")
    for row in summary.itertuples(index=False):
        print(f"  {row.scenario}: PD {row.pd_multiplier:.2f}x, LGD +{row.lgd_uplift_pp:.0f}pp; "
              f"ECL={row.ecl_total:,.2f}; change={row.change_vs_baseline:+,.2f} "
              f"({row.pct_change_vs_baseline:+.2f}%)")
    print("By-stage ECL totals:")
    print(by_stage.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"  Existing illustrative RWA remains separate/unchanged: {rwa_total:,.2f}.")
    print("Limitations: origination PDs, constant historical training-mean LGD, "
          "contractual projected EAD, contractual-rate discount proxy (not verified EIR), "
          "fixed synthetic stages, no macroeconomic calibration, and no reporting-date "
          "or regulatory stress test.")
    for path in (SUMMARY_OUT, BY_STAGE_OUT, FIGURE_OUT):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Stress report missing/empty: {path}")
        print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()