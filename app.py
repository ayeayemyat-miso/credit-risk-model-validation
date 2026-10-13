"""Local Streamlit dashboard for the educational credit-risk validation project."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

PAGES = [
    "Portfolio Overview",
    "Model Validation",
    "Illustrative ECL & RWA",
    "Stress Testing",
]

st.set_page_config(
    page_title="Credit Risk Model Validation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172b3a;
        --muted: #637381;
        --line: #dce4e9;
        --paper: #f4f7f8;
        --teal: #177e89;
        --teal-soft: #e5f2f1;
        --amber: #bd7130;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 1.6rem; }
    [data-testid="stSidebar"] { background: #eaf0f2; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #334155 !important;
    }
    [data-testid="stSidebar"] .sidebar-title {
        color: #17324D;
        font-size: 1.16rem;
        font-weight: 700;
        line-height: 1.35;
        margin: 0.55rem 0 0.85rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {
        color: #334155 !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] p {
        color: #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:hover {
        background-color: #e8eff0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:hover p {
        color: #17324D !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        background-color: #DDF3F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] p {
        color: #116466 !important;
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 3px solid var(--teal);
        border-radius: 5px;
        padding: 0.8rem 1rem;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 650; }
    h1, h2, h3 { color: var(--ink); }
    .eyebrow {
        color: var(--teal);
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .page-intro { color: #405361; max-width: 1050px; font-size: 1.02rem; }
    .source-note { color: var(--muted); font-size: 0.82rem; }
    .caveat {
        border-left: 3px solid var(--amber);
        background: #fff8ef;
        padding: 0.75rem 1rem;
        color: #563d27;
        border-radius: 3px;
        margin: 0.5rem 0 1rem 0;
    }
    div[data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_parquet(path, columns=None):
    return pd.read_parquet(path, columns=columns)


@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


def require_files(paths):
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required dashboard sources are missing: " + ", ".join(missing))


def load_data():
    required = [
        PROCESSED / "pd_scores.parquet",
        PROCESSED / "lgd_scores.parquet",
        PROCESSED / "staging.parquet",
        PROCESSED / "ecl_results.parquet",
        PROCESSED / "rwa_results.parquet",
        REPORTS / "model_validation_summary.csv",
        REPORTS / "pd_calibration_deciles.csv",
        REPORTS / "pd_calibration_subgrade.csv",
        REPORTS / "lgd_validation_segments.csv",
        REPORTS / "stress_testing_summary.csv",
        REPORTS / "stress_testing_by_stage.csv",
        FIGURES / "pd_roc_curve.png",
        FIGURES / "pd_calibration_plot.png",
        FIGURES / "stress_testing_ecl.png",
    ]
    require_files(required)

    data = {
        "pd": read_parquet(str(PROCESSED / "pd_scores.parquet")),
        "lgd": read_parquet(str(PROCESSED / "lgd_scores.parquet")),
        "staging": read_parquet(str(PROCESSED / "staging.parquet")),
        "ecl": read_parquet(str(PROCESSED / "ecl_results.parquet")),
        "rwa": read_parquet(str(PROCESSED / "rwa_results.parquet")),
        "validation": read_csv(str(REPORTS / "model_validation_summary.csv")),
        "deciles": read_csv(str(REPORTS / "pd_calibration_deciles.csv")),
        "subgrade": read_csv(str(REPORTS / "pd_calibration_subgrade.csv")),
        "lgd_segments": read_csv(str(REPORTS / "lgd_validation_segments.csv")),
        "stress": read_csv(str(REPORTS / "stress_testing_summary.csv")),
        "stress_by_stage": read_csv(str(REPORTS / "stress_testing_by_stage.csv")),
        "roc_image": FIGURES / "pd_roc_curve.png",
        "calibration_image": FIGURES / "pd_calibration_plot.png",
        "stress_image": FIGURES / "stress_testing_ecl.png",
    }
    pd_scores = data["pd"]
    if pd_scores.id.isna().any() or pd_scores.id.duplicated().any():
        raise ValueError("pd_scores must contain unique, non-null loan IDs.")
    reference_ids = pd.Index(pd_scores.id)
    for name in ("lgd", "staging", "ecl", "rwa"):
        frame = data[name]
        if frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f"{name} output has null or duplicate IDs.")
        if len(reference_ids.difference(pd.Index(frame.id))) or len(pd.Index(frame.id).difference(reference_ids)):
            raise ValueError(f"{name} IDs do not match the authoritative PD universe.")
        if not frame.set_index("id").loc[pd_scores.id, "split"].reset_index(drop=True).equals(
            pd.Series(pd_scores.split.to_numpy(), name="split")
        ):
            raise ValueError(f"{name} split values differ from pd_scores.")
        aligned = frame.set_index("id").reindex(pd_scores.id)
        if "term_months" in frame and not aligned["term_months"].reset_index(drop=True).equals(
            pd.Series(pd_scores.term_months.to_numpy(), name="term_months")
        ):
            raise ValueError(f"{name} term values differ from pd_scores.")
        if "loan_status" in frame and not aligned["loan_status"].reset_index(drop=True).equals(
            pd.Series(pd_scores.loan_status.to_numpy(), name="loan_status")
        ):
            raise ValueError(f"{name} loan statuses differ from pd_scores.")

    status_counts = pd_scores.loan_status.value_counts().to_dict()
    if sum(status_counts.values()) != len(pd_scores):
        raise ValueError("PD status counts do not cover the authoritative loan universe.")
    current_staging = data["staging"].loc[data["staging"].loan_status.eq("Current")]
    stage_counts = current_staging.synthetic_stage.value_counts().to_dict()
    allowed_stages = {"Synthetic Stage 1", "Synthetic Stage 2"}
    if (current_staging.synthetic_stage.isna().any()
            or not set(stage_counts).issubset(allowed_stages)):
        raise ValueError(f"Invalid Current synthetic-stage population: {stage_counts}.")
    if data["staging"].loc[
        ~data["staging"].loan_status.eq("Current"), "synthetic_stage"
    ].notna().any():
        raise ValueError("Non-Current rows contain synthetic stages.")

    ecl = data["ecl"]
    current_mask = ecl.loan_status.eq("Current")
    stage1_mask = current_mask & ecl.synthetic_stage.eq("Synthetic Stage 1")
    stage2_mask = current_mask & ecl.synthetic_stage.eq("Synthetic Stage 2")
    if (ecl.loc[~current_mask, "ecl_selected_synthetic"].notna().any()
            or ecl.loc[current_mask, "ecl_selected_synthetic"].isna().any()):
        raise ValueError("Selected ECL must be populated only for staged Current loans.")
    if (not ecl.loc[stage1_mask, "ecl_selected_synthetic"].equals(ecl.loc[stage1_mask, "ecl_12m"])
            or not ecl.loc[stage2_mask, "ecl_selected_synthetic"].equals(ecl.loc[stage2_mask, "ecl_lifetime"])):
        raise ValueError("ECL selected horizon does not match synthetic stage.")

    rwa = data["rwa"]
    rwa_current = rwa.loan_status.eq("Current")
    if (not rwa.included_in_current_portfolio.eq(rwa_current).all()
            or rwa.loc[~rwa_current, "rwa_illustrative"].notna().any()):
        raise ValueError("RWA inclusion does not isolate Current loans.")

    stress = data["stress"]
    stress_by_stage = data["stress_by_stage"]
    scenarios = stress.scenario.tolist()
    expected_scenarios = {"Baseline", "PD stress", "LGD stress", "Combined adverse"}
    if (len(scenarios) != 4 or stress.scenario.duplicated().any()
            or set(scenarios) != expected_scenarios
            or stress_by_stage.scenario.nunique() != 4):
        raise ValueError("Stress report must contain four unique scenarios.")
    stage_totals = stress_by_stage.groupby("scenario").ecl_total.sum().reindex(scenarios)
    summary_totals = stress.set_index("scenario").loc[scenarios, "ecl_total"]
    if not np.allclose(stage_totals.to_numpy(float), summary_totals.to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("Stress scenario totals do not reconcile to by-stage components.")
    baseline_counts = (stress_by_stage.loc[stress_by_stage.scenario.eq("Baseline")]
                       .set_index("synthetic_stage").loan_count.to_dict())
    for scenario in scenarios:
        counts = (stress_by_stage.loc[stress_by_stage.scenario.eq(scenario)]
                  .set_index("synthetic_stage").loan_count.to_dict())
        if counts != baseline_counts:
            raise ValueError(f"Stress stage membership changed in {scenario}.")
    return data


def summary_metric(data, section, metric):
    rows = data["validation"].loc[
        data["validation"].section.eq(section)
        & data["validation"].metric.eq(metric),
        "value",
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one validation summary row for {section}/{metric}.")
    value = pd.to_numeric(rows.iloc[0], errors="coerce")
    if pd.isna(value):
        raise ValueError(f"Validation summary value is not numeric: {section}/{metric}.")
    return float(value)


def fmt_count(value):
    return f"{int(value):,}"


def fmt_percent(value, digits=2):
    return f"{float(value):.{digits}%}"


def fmt_source_units(value):
    return f"{float(value):,.2f}"


def section_intro(title, intro):
    st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-intro">{intro}</div>', unsafe_allow_html=True)
    st.write("")


def page_overview(data):
    pd_scores = data["pd"]
    staging = data["staging"]
    rwa = data["rwa"]
    status_counts = pd_scores.loan_status.value_counts()
    current = pd_scores.loan_status.eq("Current")
    current_rwa = rwa.loc[rwa.included_in_current_portfolio]
    exposure = float(current_rwa.out_prncp_snapshot.sum())
    stages = staging.loc[staging.loan_status.eq("Current"), "synthetic_stage"].value_counts()

    st.markdown('<div class="eyebrow">Portfolio monitoring</div>', unsafe_allow_html=True)
    st.title("Credit Risk Model Validation Dashboard")
    st.subheader("Educational Quantitative Finance Portfolio Project")
    st.markdown(
        "This project demonstrates an end-to-end credit risk modelling and validation "
        "workflow using historical LendingClub consumer loan data. It brings together "
        "PD, LGD, EAD, illustrative ECL and RWA, model validation, and sensitivity analysis "
        "to demonstrate quantitative modelling and transparent communication of limitations."
    )
    st.markdown(
        '<div class="caveat"><strong>Educational portfolio project.</strong> '
        'This is not a production model, accounting allowance, or regulatory reporting system.</div>',
        unsafe_allow_html=True,
    )

    section_intro(
        "Portfolio Overview",
        "This section summarizes the historical/modelled population and separates it "
        "from the smaller set of loans currently recorded as Current. That distinction "
        "matters because historical outcomes are not live portfolio exposures.",
    )
    first = st.columns(4)
    first[0].metric("Modelled loans", fmt_count(len(pd_scores)))
    first[1].metric("Fully Paid", fmt_count(status_counts.get("Fully Paid", 0)))
    first[2].metric("Charged Off", fmt_count(status_counts.get("Charged Off", 0)))
    first[3].metric("Current", fmt_count(int(current.sum())))
    second = st.columns(4)
    second[0].metric("Current recorded exposure", fmt_source_units(exposure))
    second[1].metric("Synthetic Stage 1", fmt_count(stages.get("Synthetic Stage 1", 0)))
    second[2].metric("Synthetic Stage 2", fmt_count(stages.get("Synthetic Stage 2", 0)))
    second[3].metric("Out-of-time test loans", fmt_count(pd_scores.split.eq("test").sum()))
    st.caption("Exposure is shown in source-currency units; currency is not identified in the input data.")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Loan status by cohort")
        by_split = pd.crosstab(pd_scores.split, pd_scores.loan_status)
        st.bar_chart(by_split)
        st.caption("Train covers 2011–2013 vintages; test is the 2014 out-of-time cohort.")
    with right:
        st.subheader("Current synthetic staging")
        stage_frame = (staging.loc[staging.loan_status.eq("Current"), "synthetic_stage"]
                       .value_counts().rename_axis("Synthetic stage").rename("Loans").reset_index())
        st.dataframe(stage_frame, hide_index=True, use_container_width=True)
        st.caption("Synthetic Stage 1/2 are illustrative scenario labels, not observed SICR assessments.")

    with st.expander("Understanding the Dataset and Modelling Workflow"):
        st.markdown(
            "A loan portfolio is a set of exposures whose credit losses and capital needs "
            "are monitored together. **Fully Paid** and **Charged Off** rows are historical "
            "outcomes; **Current** rows are the only loans represented as live in the "
            "illustrative ECL/RWA summaries."
        )
        st.markdown(
            "The train/test split keeps 2011–2013 loans for model fitting and 2014 loans "
            "for out-of-time evaluation. Testing on a later vintage is useful because it "
            "checks performance on a subsequent period, but the deterministic sample is "
            "stratified by loan status, term, year, and split."
        )
        st.markdown(
            "Workflow: a discrete-time hazard PD model produces origination-based PD; "
            "LGD uses realized recovery targets and a selected training-mean baseline; "
            "EAD is a contractual amortization schedule; ECL and RWA are explicitly "
            "illustrative calculations. Synthetic staging is not a measured reporting-date "
            "classification."
        )


def page_validation(data):
    section_intro(
        "Model Validation",
        "This section assesses PD ranking and calibration on the out-of-time test loans "
        "with sufficient follow-up, and evaluates LGD only where a valid realized target "
        "exists for a test-split charge-off.",
    )
    validation = data["validation"]
    pd_section = "PD 12-month test"
    auc = summary_metric(data, pd_section, "auc")
    gini = summary_metric(data, pd_section, "gini")
    ks = summary_metric(data, pd_section, "ks")
    brier = summary_metric(data, pd_section, "brier")
    mean_pd = summary_metric(data, pd_section, "mean_predicted_pd")
    observed = summary_metric(data, pd_section, "observed_event_rate")
    eligible = int(summary_metric(data, pd_section, "eligible_test_loans"))
    events = int(summary_metric(data, pd_section, "events"))
    excluded = int(summary_metric(data, pd_section, "excluded_test_loans"))

    st.subheader("Probability of Default")
    metrics_a = st.columns(4)
    metrics_a[0].metric("ROC-AUC", f"{auc:.4f}", help="Rank-based ability to distinguish events from non-events.")
    metrics_a[1].metric("Gini", f"{gini:.4f}", help="Calculated as 2 × AUC − 1.")
    metrics_a[2].metric("KS", f"{ks:.4f}", help="Maximum separation between event and non-event score distributions.")
    metrics_a[3].metric("Brier score", f"{brier:.5f}", help="Mean squared error of predicted event probabilities; lower is better.")
    metrics_b = st.columns(4)
    metrics_b[0].metric("Mean predicted 12-month PD", fmt_percent(mean_pd))
    metrics_b[1].metric("Observed event rate", fmt_percent(observed))
    metrics_b[2].metric("Eligible test loans", fmt_count(eligible))
    metrics_b[3].metric("Events / excluded", f"{fmt_count(events)} / {fmt_count(excluded)}")
    st.caption(
        f"Eligible out-of-time 2014 test subset: {fmt_count(eligible)} loans; "
        f"{fmt_count(events)} events; {fmt_count(excluded)} test loans excluded. "
        "Observed defaults use the six-month charge-off timing proxy."
    )
    st.markdown(
        f"On this eligible subset, mean predicted PD is {fmt_percent(mean_pd)} and the "
        f"observed event rate is {fmt_percent(observed)}. AUC {auc:.3f} indicates some "
        "ranking ability; calibration and outcome-selection limitations still matter."
    )

    roc_col, calibration_col = st.columns(2)
    with roc_col:
        st.image(data["roc_image"], use_container_width=True)
        st.caption("ROC curve uses eligible out-of-time test loans only.")
    with calibration_col:
        st.image(data["calibration_image"], use_container_width=True)
        st.caption("Calibration plot uses eligible out-of-time test loans only.")

    with st.expander("Understanding PD Validation Metrics"):
        st.markdown(
            "**ROC-AUC** measures ranking across thresholds. **Gini = 2 × AUC − 1** is "
            "a rescaling of AUC. **KS** is the maximum TPR–FPR separation. These are "
            "discrimination measures; they do not establish calibration. **Brier score** "
            "is the average squared difference between a predicted probability and the "
            "binary outcome."
        )
        st.markdown(
            "Observed-versus-predicted rates compare probability level to event frequency. "
            "The decile table and calibration plot show this relationship in groups; small "
            "differences should be interpreted with group counts and sampling uncertainty in mind."
        )
    with st.expander("PD Test Cohort and Limitations"):
        st.markdown(
            "The endpoint is `event_12m = is_event == 1 and event_month <= 12`; an "
            "event-free loan is eligible as a non-event only when `obs_end >= 12`. "
            "This is a complete-follow-up subset, not the full test cohort. Earlier "
            "payoffs are competing outcomes and early censoring is excluded, so the "
            "metrics are not an IPCW or competing-risk estimate of full-population calibration."
        )
        st.markdown(
            "The default date is proxied by last payment month plus six months. PDs are "
            "origination-based estimates, not reporting-date or regulatory long-run PDs. "
            "The 15% sample is stratified by loan status, term, year, and split."
        )

    st.subheader("Loss Given Default")
    lgd_data = data["validation"]
    get_lgd = lambda metric: summary_metric(data, "LGD test", metric)
    lgd_metrics = st.columns(4)
    lgd_metrics[0].metric("Valid test charge-offs", fmt_count(get_lgd("n")))
    lgd_metrics[1].metric("MAE", f"{get_lgd('mae'):.4f}")
    lgd_metrics[2].metric("RMSE", f"{get_lgd('rmse'):.4f}")
    lgd_metrics[3].metric("R-squared", f"{get_lgd('r2'):.4f}")
    lgd_metrics2 = st.columns(4)
    lgd_metrics2[0].metric("Mean realized LGD", fmt_percent(get_lgd("mean_realised")))
    lgd_metrics2[1].metric("Mean predicted LGD", fmt_percent(get_lgd("mean_predicted")))
    lgd_metrics2[2].metric("Prediction bias", f"{get_lgd('bias'):+.4f}")
    lgd_metrics2[3].metric("Exact-1 realized share", fmt_percent(get_lgd("exact_one_share")))
    st.markdown(
        "The selected prediction is a constant training-mean LGD baseline. The negative "
        "test R-squared indicates it does not explain test-target variation; it is not a "
        "differentiated or regulatory downturn LGD model."
    )
    st.dataframe(
        data["lgd_segments"].style.format({
            "mean_realised_lgd": "{:.2%}", "mean_predicted_lgd": "{:.2%}",
            "bias_predicted_minus_realised": "{:+.4f}", "mae": "{:.4f}",
            "rmse": "{:.4f}", "r2": "{:.4f}", "exact_one_share": "{:.1%}",
        }, na_rep="-"),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Understanding LGD Validation Metrics"):
        st.markdown(
            "LGD is loss severity conditional on default. MAE is mean absolute error; "
            "RMSE penalizes larger errors more; bias is predicted minus realized LGD; "
            "R-squared compares error with the target variance. Errors are measured on "
            "the 0–1 LGD fraction scale. Results are conditional on Charged Off loans "
            "with valid realized targets."
        )
    with st.expander("LGD Model Limitations"):
        st.markdown(
            "The realized target uses recoveries and collection fees divided by a "
            "scheduled-amortization EAD proxy at last payment plus six months. It is not "
            "a verified charge-off balance. Of 11,073 charge-offs, 741 were excluded for "
            "nonpositive proxy EAD; 156 raw negative LGDs were clipped to zero. Recovery "
            "maturity and vintage differences affect test comparability."
        )


def page_ecl_rwa(data):
    section_intro(
        "Illustrative ECL & RWA",
        "This section presents selected synthetic ECL for Current loans and a separate "
        "hypothetical exposure-times-risk-weight illustration. ECL estimates expected "
        "loss; RWA is a distinct capital exposure measure, so the values are not reconciliations.",
    )
    ecl = data["ecl"]
    current_ecl = ecl.loc[ecl.loan_status.eq("Current")].copy()
    stage1 = current_ecl.synthetic_stage.eq("Synthetic Stage 1")
    stage2 = current_ecl.synthetic_stage.eq("Synthetic Stage 2")
    if current_ecl.ecl_selected_synthetic.isna().any():
        raise ValueError("Current selected synthetic ECL contains missing values.")
    total_ecl = float(current_ecl.ecl_selected_synthetic.sum())
    stage1_ecl = float(current_ecl.loc[stage1, "ecl_selected_synthetic"].sum())
    stage2_ecl = float(current_ecl.loc[stage2, "ecl_selected_synthetic"].sum())

    rwa = data["rwa"]
    current_rwa = rwa.loc[rwa.included_in_current_portfolio]
    if not current_rwa.loan_status.eq("Current").all():
        raise ValueError("RWA included population contains non-Current loans.")
    exposure = float(current_rwa.out_prncp_snapshot.sum())
    rwa_baseline = float(current_rwa.rwa_illustrative.sum())
    if abs(rwa_baseline - exposure) >= 0.01:
        raise ValueError("Illustrative 100% RWA does not reconcile to Current exposure.")
    risk_weight_sensitivities = pd.DataFrame({
        "hypothetical_risk_weight": [0.75, 1.00, 1.25, 1.50],
    })
    risk_weight_sensitivities["illustrative_rwa"] = (
        exposure * risk_weight_sensitivities["hypothetical_risk_weight"]
    )
    risk_weight_sensitivities["assumption"] = "Hypothetical only; not applicable Basel/CRR risk weights"

    metrics = st.columns(4)
    metrics[0].metric("Current selected synthetic ECL", fmt_source_units(total_ecl))
    metrics[1].metric("Stage 1 selected ECL", fmt_source_units(stage1_ecl))
    metrics[2].metric("Stage 2 selected ECL", fmt_source_units(stage2_ecl))
    metrics[3].metric("Current recorded exposure", fmt_source_units(exposure))
    st.caption("Amounts are in source-currency units; the input data do not establish EUR.")

    left, right = st.columns(2)
    with left:
        st.subheader("Selected ECL by synthetic stage")
        stage_table = pd.DataFrame({
            "Synthetic stage": ["Synthetic Stage 1", "Synthetic Stage 2"],
            "Loans": [int(stage1.sum()), int(stage2.sum())],
            "Selected ECL": [stage1_ecl, stage2_ecl],
        })
        st.bar_chart(stage_table.set_index("Synthetic stage")[["Selected ECL"]])
        st.dataframe(stage_table.style.format({"Selected ECL": "{:,.2f}"}), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Hypothetical RWA sensitivity")
        st.metric("Illustrative RWA at 100% assumed weight", fmt_source_units(rwa_baseline))
        st.dataframe(
            risk_weight_sensitivities.style.format({
                "hypothetical_risk_weight": "{:.0%}", "illustrative_rwa": "{:,.2f}",
            }),
            hide_index=True,
            use_container_width=True,
        )
    st.markdown(
        "The RWA illustration applies a hypothetical 100% weight to the recorded Current "
        "outstanding principal. The 75%, 100%, 125%, and 150% values are scenario "
        "assumptions, not verified regulatory weights."
    )

    with st.expander("How ECL Is Calculated"):
        st.markdown(
            "Conceptually, expected loss combines default likelihood, loss severity, and "
            "exposure: **Expected Loss ≈ PD × LGD × EAD**. This project uses monthly "
            "conditional hazards converted to marginal default probabilities with survival, "
            "then multiplies monthly marginal PD × baseline LGD × contractual start-of-month "
            "EAD × a discount factor. Stage 1 selects the first 12 months; synthetic Stage 2 "
            "selects the contractual lifetime."
        )
        st.markdown(
            "The simplified formula does not replace monthly timing. Discounting uses "
            "contractual `int_rate` as a proxy, not a verified effective interest rate. "
            "The EAD schedule is projected from origination and assumes contractual "
            "amortization, not observed reporting-date exposure."
        )
    with st.expander("Understanding Hypothetical RWA"):
        st.markdown(
            "RWA represents exposure after applying regulatory risk weights and supports "
            "capital requirements. The displayed calculation is only recorded Current "
            "principal × hypothetical weight. Regulatory exposure class, eligible "
            "collateral, guarantees, CCF, applicable framework treatment, and supervisory "
            "approval are not established here. ECL and RWA use different methodologies."
        )
    with st.expander("Assumptions and Limitations"):
        st.markdown(
            "Only Current loans contribute to the selected portfolio summaries. Fully Paid "
            "and Charged Off loans remain historical records, not live exposures. Synthetic "
            "Stage 1/2 labels do not represent observed reporting-date IFRS 9 SICR. The 45% "
            "LGD field is a scenario benchmark, not an established portfolio requirement. "
            "All amounts are source-currency units; currency is not specified."
        )


def page_stress_testing(data):
    section_intro(
        "Stress Testing",
        "This page compares four saved educational sensitivities for Current synthetic "
        "Stage 1/2 loans. PD and LGD are shocked separately and jointly while stage "
        "membership is held fixed, so the comparison isolates those assumptions.",
    )
    stress = data["stress"]
    by_stage = data["stress_by_stage"]
    expected_order = ["Baseline", "PD stress", "LGD stress", "Combined adverse"]
    if stress.scenario.tolist() != expected_order:
        raise ValueError(f"Unexpected stress scenario order: {stress.scenario.tolist()}.")
    combined = stress.loc[stress.scenario.eq("Combined adverse")].iloc[0]
    baseline = stress.loc[stress.scenario.eq("Baseline")].iloc[0]
    if (stress.loan_count.nunique() != 1 or int(stress.loan_count.iloc[0]) != 1_789):
        raise ValueError("Stress summary does not consistently cover 1,789 Current loans.")
    stage_counts = (by_stage.loc[by_stage.scenario.eq("Baseline")]
                    .set_index("synthetic_stage").loan_count.to_dict())
    if stage_counts != {"Synthetic Stage 1": 1_275, "Synthetic Stage 2": 514}:
        raise ValueError(f"Stress stage counts differ from staging source: {stage_counts}.")

    metrics = st.columns(3)
    metrics[0].metric("Baseline selected ECL", fmt_source_units(baseline.ecl_total))
    metrics[1].metric("Combined adverse ECL", fmt_source_units(combined.ecl_total))
    metrics[2].metric("Combined change", f"{combined.pct_change_vs_baseline:+.2f}%",
                      delta=fmt_source_units(combined.change_vs_baseline))
    st.caption("All scenario values are selected synthetic ECL for the Current portfolio, in source-currency units.")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.image(data["stress_image"], use_container_width=True)
        st.caption("Existing stress-testing figure; fixed Synthetic Stage 1/2 contributions.")
    with right:
        st.subheader("Scenario assumptions and totals")
        display = stress[[
            "scenario", "pd_multiplier", "lgd_uplift_pp", "ecl_total",
            "change_vs_baseline", "pct_change_vs_baseline",
        ]].copy()
        st.dataframe(display.style.format({
            "pd_multiplier": "{:.2f}x",
            "lgd_uplift_pp": "+{:.0f} pp",
            "ecl_total": "{:,.2f}",
            "change_vs_baseline": "+{:,.2f}",
            "pct_change_vs_baseline": "+{:.2f}%",
        }), hide_index=True, use_container_width=True)
        st.caption(
            f"Combined adverse increases selected ECL by "
            f"{fmt_source_units(combined.change_vs_baseline)} "
            f"({combined.pct_change_vs_baseline:.2f}%) versus baseline under these assumptions."
        )

    st.subheader("Stress contribution by synthetic stage")
    stage_view = by_stage.copy()
    stage_view["ECL contribution"] = stage_view["ecl_total"]
    st.bar_chart(
        stage_view.pivot(index="scenario", columns="synthetic_stage", values="ECL contribution")
        .reindex(expected_order)
    )
    st.dataframe(by_stage.style.format({
        "ecl_total": "{:,.2f}",
        "change_vs_baseline": "+{:,.2f}",
        "pct_change_vs_baseline": "+{:.2f}%",
    }), hide_index=True, use_container_width=True)

    with st.expander("Understanding the Stress Scenarios"):
        st.markdown(
            "Stress testing examines how results respond to adverse assumptions. Here, "
            "the baseline is unchanged; PD stress targets 1.25× cumulative PD; LGD stress "
            "adds 10 percentage points capped at 100%; combined adverse applies 1.50× "
            "cumulative PD and the same LGD uplift. These are not macroeconomically calibrated forecasts."
        )
    with st.expander("How Stressed ECL Is Calculated"):
        st.markdown(
            "Monthly conditional hazards are first converted to marginal default "
            "probabilities with survival. Marginal probabilities are proportionally "
            "scaled to the capped cumulative-PD target, and conditional hazards can be "
            "reconstructed from stressed survival for validation. This avoids multiplying "
            "conditional hazards directly and double-counting default risk."
        )
        st.markdown(
            "LGD is increased uniformly because the baseline prediction is a constant "
            "training-mean value. Synthetic stage membership does not migrate under stress; "
            "Stage 1 continues to select 12-month ECL and Stage 2 lifetime ECL."
        )
    with st.expander("Stress Testing Limitations"):
        st.markdown(
            "The scenarios use origination-based PDs, historical constant LGD, projected "
            "contractual EAD, and contractual-rate discounting rather than verified EIR. "
            "They do not model reporting-date SICR, stage migration, macroeconomic paths, "
            "or regulatory stress-test requirements. RWA is separate and unchanged."
        )


def main():
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Dashboard data validation failed: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="eyebrow">Risk analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Credit Risk Validation</div>', unsafe_allow_html=True)
        page = st.radio("Dashboard section", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Historical LendingClub sample · local read-only dashboard")
        st.caption("Educational analysis; not regulatory reporting.")

    if page == "Portfolio Overview":
        page_overview(data)
    elif page == "Model Validation":
        page_validation(data)
    elif page == "Illustrative ECL & RWA":
        page_ecl_rwa(data)
    elif page == "Stress Testing":
        page_stress_testing(data)


if __name__ == "__main__":
    main()