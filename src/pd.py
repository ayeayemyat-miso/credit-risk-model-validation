"""Phase 2: LendingClub origination PD via discrete-time logistic hazards.

Run from project root: python src/pd.py
Input: data/processed/loans_clean.parquet
Output: data/processed/pd_scores.parquet

Six-month lag after last payment is a *proxy* for charge-off timing, not a
verified default date. Fully Paid loans are censored at final payment (a
simplifying assumption; prepayment is a competing risk). Predictions are
origination-based, not reporting-date IFRS 9 conditional/forward-looking PDs.
"""
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / 'data' / 'processed' / 'loans_clean.parquet'
OUT = ROOT / 'data' / 'processed' / 'pd_scores.parquet'
HAZARD_CURVES_OUT = ROOT / 'data' / 'processed' / 'hazard_curves.parquet'
RANDOM_SEED = 42
DEFAULT_LAG_MONTHS = 6
SUBSAMPLE_FRAC = 0.15  # set to 1.0 for full training and evaluation
CHUNK_SIZE = 2_000
TERMS = (36, 60)

NUMERIC_FEATURES = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment',
    'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
    'total_il_high_credit_limit', 'avg_cur_bal', 'bc_open_to_buy',
    'bc_util', 'acc_open_past_24mths', 'num_actv_bc_tl',
    'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
    'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
    'num_sats', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mths_since_recent_bc', 'mths_since_recent_inq',
    'mths_since_recent_revol_delinq', 'total_rev_hi_lim',
    'all_util', 'il_util', 'max_bal_bc', 'open_acc_6m',
    'open_act_il', 'open_il_12m', 'open_il_24m',
    'mths_since_rcnt_il', 'total_bal_il', 'open_rv_12m',
    'open_rv_24m', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
    'tot_coll_amt', 'tot_cur_bal', 'acc_now_delinq',
    'delinq_amnt', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m',
    'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m', 'chargeoff_within_12_mths',
    'mths_since_recent_bc_dlq', 'mths_since_last_major_derog',
    'collections_12_mths_ex_med',
]
CATEGORICAL_FEATURES = [
    'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state', 'application_type',
]
EVENT_NULL_FEATURES = [
    'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'mths_since_recent_bc_dlq',
    'mths_since_recent_revol_delinq',
]


def load_and_prepare():
    print('Loading cleaned data ...')
    df = pd.read_parquet(IN)
    print(f'  Original loans: {len(df):,}')
    needed = ['id', 'loan_status', 'issue_year', 'split', 'term_months',
              'months_to_last_pymnt', 'months_seasoned', 'default', 'sub_grade']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in loans_clean.parquet: {missing}')
    if df['id'].isna().any() or df['id'].duplicated().any():
        raise ValueError('Loan id must be non-null and unique.')
    if not df['split'].isin(['train', 'test']).all():
        raise ValueError("split must contain only 'train' or 'test'.")
    df = df[df['term_months'].isin(TERMS)].copy()
    if SUBSAMPLE_FRAC < 1:
        if not 0 < SUBSAMPLE_FRAC <= 1:
            raise ValueError('SUBSAMPLE_FRAC must be in (0, 1].')
        # Preserve the split, term, year, and status distributions for debugging.
        df = df.groupby(['split', 'term_months', 'issue_year', 'loan_status'],
                        dropna=False, group_keys=False).sample(
                            frac=SUBSAMPLE_FRAC, random_state=RANDOM_SEED)
    df = df.reset_index(drop=True)
    print(f'  Selected loans: {len(df):,}; train={(df.split == "train").sum():,}; '
          f'test={(df.split == "test").sum():,}')
    num_cols = [c for c in NUMERIC_FEATURES if c in df]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df]
    if not num_cols or not cat_cols:
        raise ValueError('No usable numeric or categorical features found.')
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    flags = []
    for c in EVENT_NULL_FEATURES:
        if c in num_cols:
            name = c + '_missing'
            df[name] = df[c].isna().astype(np.float32)
            df[c] = df[c].fillna(999.0)
            flags.append(name)
    train = df['split'].eq('train')
    for c in num_cols:
        median = df.loc[train, c].median()
        df[c] = df[c].fillna(float(median) if pd.notna(median) else 0.0).astype(np.float32)
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
    return df, num_cols + flags, cat_cols


def observation_metadata(df):
    """End at observed event, censoring, snapshot, or contractual term.

    Do not turn a charge-off proxy after snapshot/term into an event at maturity.
    """
    m = df[['id', 'loan_status', 'term_months', 'months_to_last_pymnt',
            'months_seasoned', 'split', 'default', 'issue_year', 'sub_grade']].copy()
    last = pd.to_numeric(m['months_to_last_pymnt'], errors='coerce')
    seasoned = pd.to_numeric(m['months_seasoned'], errors='coerce')
    term = pd.to_numeric(m['term_months'], errors='coerce')
    charge = m['loan_status'].eq('Charged Off')
    paid = m['loan_status'].eq('Fully Paid')
    current = m['loan_status'].eq('Current')
    if not (charge | paid | current).all():
        raise ValueError('Unexpected loan statuses; update observation rules first.')
    event_month = last + DEFAULT_LAG_MONTHS
    m['event_month'] = event_month.where(charge)
    # Observed follow-up cannot extend beyond the dataset snapshot.
    # For fully paid loans, censor at last payment; for current at snapshot.
    raw_end = np.where(charge, event_month, np.where(paid, last, seasoned))
    end = np.minimum(np.minimum(raw_end, seasoned), term)
    m['obs_end'] = np.floor(end)
    m['is_event'] = (charge & event_month.notna() & (event_month >= 1)
                     & (event_month <= seasoned) & (event_month <= term)).astype(np.int8)
    # A Charged Off label with a missing last payment date cannot establish
    # a valid event time; omit it rather than inventing one.
    valid = m['obs_end'].notna() & (m['obs_end'] >= 1)
    removed = int((~valid).sum())
    if removed:
        print(f'  Excluding {removed:,} loans without a usable observation interval')
    m = m.loc[valid].copy()
    m['obs_end'] = m['obs_end'].astype(np.int16)
    return m


def expand_panel(loans, durations):
    """Expand minimal feature frame; keep unique row indices for encoding."""
    repeat = np.repeat(np.arange(len(loans)), durations.astype(int).to_numpy())
    panel = loans.iloc[repeat].reset_index(drop=True)
    panel['month'] = np.arange(len(panel), dtype=np.int32) - np.repeat(
        np.cumsum(durations.to_numpy(dtype=np.int64)) - durations.to_numpy(dtype=np.int64),
        durations.to_numpy(dtype=np.int64)) + 1
    panel['month'] = panel['month'].astype(np.int16)
    return panel


def make_encoder():
    # sklearn >=1.2 uses sparse_output; earlier versions use sparse.
    try:
        return OneHotEncoder(handle_unknown='ignore', drop='first',
                             sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', drop='first',
                             sparse=True, dtype=np.float32)


def fit_transformers(train_panel, num_cols, cat_cols, term):
    scaler = StandardScaler()
    scaler.fit(train_panel[num_cols])
    cat_enc = make_encoder()
    cat_enc.fit(train_panel[cat_cols])
    # Fix the month baseline at month 1 even when some months are absent.
    month_enc = OneHotEncoder(categories=[np.arange(1, term + 1)],
                             handle_unknown='ignore', drop='first',
                             **({'sparse_output': True} if 'sparse_output' in
                                __import__('inspect').signature(OneHotEncoder).parameters
                                else {'sparse': True}), dtype=np.float32)
    month_enc.fit(pd.DataFrame({'month': np.arange(1, term + 1)}))
    return scaler, cat_enc, month_enc


def transform(panel, num_cols, cat_cols, transformers):
    scaler, cat_enc, month_enc = transformers
    x_num = sparse.csr_matrix(scaler.transform(panel[num_cols]).astype(np.float32))
    x_cat = cat_enc.transform(panel[cat_cols])
    x_mon = month_enc.transform(panel[['month']])
    return sparse.hstack([x_num, x_cat, x_mon], format='csr', dtype=np.float32)


def fit_models(df, meta, num_cols, cat_cols):
    models = {}
    features = ['id', 'split'] + num_cols + cat_cols
    for term in TERMS:
        print(f'\nTraining {term}-month hazard model ...')
        selected = meta[(meta.term_months == term) & (meta.split == 'train')]
        if selected.empty:
            raise ValueError(f'No training loans for term {term}.')
        loans = selected[['id', 'obs_end', 'is_event']].merge(
            df[features].drop(columns='split'), on='id', validate='one_to_one')
        panel = expand_panel(loans[['id', 'is_event'] + num_cols + cat_cols], loans.obs_end)
        panel['event'] = ((panel.month.to_numpy() ==
                           np.repeat(loans.obs_end.to_numpy(), loans.obs_end.to_numpy())) &
                          (panel.is_event.to_numpy() == 1)).astype(np.int8)
        if panel.event.nunique() != 2:
            raise ValueError(f'Term {term}: both event and non-event months required.')
        transformers = fit_transformers(panel, num_cols, cat_cols, term)
        X = transform(panel, num_cols, cat_cols, transformers)
        y = panel.event.to_numpy()
        model = LogisticRegression(C=1.0, max_iter=1500,
                                   solver='lbfgs', random_state=RANDOM_SEED)
        model.fit(X, y)
        print(f'  Loans: {len(loans):,}; rows: {len(panel):,}; events: {int(y.sum()):,}')
        print(f'  Training loan-month AUC (diagnostic): {roc_auc_score(y, model.predict_proba(X)[:, 1]):.4f}')
        models[term] = (model, transformers)
        del loans, panel, X, y
        gc.collect()
    return models


def score_loans(df, meta, num_cols, cat_cols, models):
    results = []
    model_curve_parts = []
    hazard_count = 0
    extreme_hazard_count = 0
    feature_df = df[['id'] + num_cols + cat_cols]
    for term in TERMS:
        selected = meta.loc[meta.term_months == term,
                            ['id', 'term_months', 'split', 'obs_end']]
        model, transformers = models[term]
        cat_enc = transformers[1]
        unknown_counts = {c: 0 for c in cat_cols}
        unknown_values = {c: set() for c in cat_cols}
        test_loan_count = 0
        print(f'\nScoring {len(selected):,} loans over {term} months ...')
        for start in range(0, len(selected), CHUNK_SIZE):
            batch = selected.iloc[start:start + CHUNK_SIZE].merge(
                feature_df, on='id', validate='one_to_one')
            test_mask = batch.split.eq('test').to_numpy()
            test_loan_count += int(test_mask.sum())
            for col, known_categories in zip(cat_cols, cat_enc.categories_):
                test_values = batch.loc[test_mask, col]
                unknown = ~test_values.isin(known_categories)
                unknown_counts[col] += int(unknown.sum())
                unknown_values[col].update(test_values.loc[unknown].unique())
            durations = batch.term_months.astype(int)
            panel = expand_panel(batch[['id'] + num_cols + cat_cols], durations)
            X = transform(panel, num_cols, cat_cols, transformers)
            hazard = model.predict_proba(X)[:, 1]
            hazard_count += len(hazard)
            extreme_hazard_count += int(np.count_nonzero(hazard >= 0.999999))
            # Rows are contiguous per loan, exactly 'term' rows each.
            hazard_matrix = hazard.reshape(-1, term)
            log_surv = np.log1p(-np.clip(hazard_matrix, 0.0,
                                         np.nextafter(1.0, 0.0)))
            results.append(pd.DataFrame({
                'id': batch.id.to_numpy(),
                'pd_12m': -np.expm1(log_surv[:, :12].sum(axis=1)),
                'pd_lifetime': -np.expm1(log_surv.sum(axis=1)),
            }))
            if test_mask.any():
                test_obs_end = batch.loc[test_mask, 'obs_end'].to_numpy()
                test_hazards = hazard_matrix[test_mask]
                for month in range(1, term + 1):
                    at_risk = test_obs_end >= month
                    model_curve_parts.append({
                        'term_months': term,
                        'month': month,
                        'model_hazard_sum': float(test_hazards[at_risk, month - 1].sum()),
                        'model_at_risk': int(at_risk.sum()),
                    })
            del panel, X, hazard, log_surv
            gc.collect()
        print('  Test loans with unknown categories (train-fitted encoders):')
        for col in cat_cols:
            count = unknown_counts[col]
            frequency = count / test_loan_count if test_loan_count else 0.0
            print(f'    {col}: {count:,}/{test_loan_count:,} ({frequency:.2%}); '
                  f'{len(unknown_values[col]):,} distinct unseen values')
    if not results:
        raise ValueError('No loans were scored.')
    print(f'  Scored loan-month hazards >= 0.999999: '
          f'{extreme_hazard_count:,}/{hazard_count:,}')
    model_curves = pd.DataFrame(model_curve_parts)
    model_curves = (model_curves.groupby(['term_months', 'month'], as_index=False)
                    .agg(model_hazard_sum=('model_hazard_sum', 'sum'),
                         model_at_risk=('model_at_risk', 'sum')))
    model_curves['model_implied_hazard'] = model_curves.model_hazard_sum.div(
        model_curves.model_at_risk.where(model_curves.model_at_risk.gt(0)))
    return pd.concat(results, ignore_index=True), model_curves


def evaluate_horizon(out, horizon, pred_col):
    """Binary metrics only for loans whose horizon outcome is known.

    An event within horizon is known positive. Non-events need at least
    horizon months of observed follow-up to be known negative. Loans
    censored early are excluded; this is NOT IPCW validation.
    """
    event_in_horizon = (out.is_event.eq(1) & out.event_month.le(horizon))
    eligible = event_in_horizon | out.obs_end.ge(horizon)
    print(f'\n{horizon}-month validation (complete-horizon subset; not IPCW)')
    for split in ('train', 'test'):
        d = out.loc[eligible & out.split.eq(split)]
        y = event_in_horizon.loc[d.index].astype(int)
        if horizon == 60 and split == 'test':
            excluded = out.loc[~eligible & out.split.eq(split)]
            print(f'  test eligibility: {len(d):,} eligible; {int(y.sum()):,} defaults; '
                  f'{int((1 - y).sum()):,} nondefaults; {len(excluded):,} excluded')
            if len(excluded):
                print('  test exclusions by status: '
                      f'{excluded.loan_status.value_counts(dropna=False).to_dict()}')
        if len(d) < 2 or y.nunique() < 2:
            if y.nunique() < 2 and len(d):
                print(f'  {split}: AUC unavailable; eligible outcomes contain '
                      f'{int(y.sum()):,} defaults and {int((1 - y).sum()):,} nondefaults '
                      '(AUC requires both classes).')
            else:
                print(f'  {split}: insufficient eligible outcomes ({len(d):,} loans)')
            continue
        auc = roc_auc_score(y, d[pred_col])
        fpr, tpr, _ = roc_curve(y, d[pred_col])
        print(f'  {split}: n={len(d):,}, AUC={auc:.4f}, Gini={2*auc-1:.4f}, '
              f'KS={np.max(tpr-fpr):.4f}, mean PD={d[pred_col].mean():.2%}, '
              f'event rate={y.mean():.2%}')


def report_36m_calibration(out):
    horizon = 36
    event = out.is_event.eq(1) & out.event_month.le(horizon)
    eligible = event | out.obs_end.ge(horizon)
    d = out.loc[eligible & out.split.eq('test') & out.term_months.eq(horizon)].copy()
    d['observed_default'] = event.loc[d.index].astype(np.int8)
    print('\n36-month test calibration (complete-horizon eligible subset; descriptive only)')
    print(f'  Eligible loans: {len(d):,}; defaults={int(d.observed_default.sum()):,}; '
          f'nondefaults={int((1 - d.observed_default).sum()):,}')

    def print_calibration(label, grouped):
        summary = grouped.agg(
            loans=('id', 'size'),
            defaults=('observed_default', 'sum'),
            predicted_pd=('pd_lifetime', 'mean'),
            observed_rate=('observed_default', 'mean'),
        )
        summary['nondefaults'] = summary.loans - summary.defaults
        summary = summary[['loans', 'defaults', 'nondefaults',
                           'predicted_pd', 'observed_rate']]
        print(f'  By {label}:')
        print(summary.to_string(float_format=lambda value: f'{value:.4f}'))

    print_calibration('sub_grade', d.groupby('sub_grade', dropna=False))
    d = d.sort_values('id').copy()
    d['pd_decile'] = pd.qcut(d.pd_lifetime.rank(method='first'), 10,
                             labels=False) + 1
    print_calibration('PD decile', d.groupby('pd_decile', observed=True))

    print('  Late-month at-risk observations (months 31-36):')
    for month in range(31, 37):
        at_risk = d.obs_end.ge(month)
        events = d.is_event.eq(1) & d.obs_end.eq(month)
        print(f'    month {month}: at risk={int(at_risk.sum()):,}; '
              f'proxy defaults={int((at_risk & events).sum()):,}')


def make_hazard_curves(meta, model_curves):
    rows = []
    test = meta.loc[meta.split.eq('test')]
    for term in TERMS:
        loans = test.loc[test.term_months.eq(term)]
        for month in range(1, term + 1):
            at_risk = loans.obs_end.ge(month)
            risk_count = int(at_risk.sum())
            event_count = int((loans.is_event.eq(1) & loans.obs_end.eq(month)).sum())
            rows.append({
                'term_months': term,
                'month': month,
                'test_at_risk': risk_count,
                'observed_defaults': event_count,
                'empirical_hazard': event_count / risk_count if risk_count else np.nan,
            })
    curves = pd.DataFrame(rows).merge(
        model_curves[['term_months', 'month', 'model_at_risk',
                      'model_implied_hazard']],
        on=['term_months', 'month'], how='left', validate='one_to_one')
    if not curves.test_at_risk.eq(curves.model_at_risk).all():
        raise AssertionError('Model and empirical hazard curves use different risk sets.')
    return curves


def main():
    df, num_cols, cat_cols = load_and_prepare()
    meta = observation_metadata(df)
    print(f'  Usable loans: {len(meta):,}; observed proxy events: {meta.is_event.sum():,}')
    models = fit_models(df, meta, num_cols, cat_cols)
    pd_all, model_curves = score_loans(df, meta, num_cols, cat_cols, models)
    out = meta.merge(pd_all, on='id', validate='one_to_one')
    if out.id.isna().any() or out.id.duplicated().any():
        raise AssertionError('Scored loan IDs must be non-null and unique.')
    if not out.term_months.isin(TERMS).all():
        raise AssertionError('Scored loans contain an invalid contractual horizon.')
    if not out.obs_end.between(1, out.term_months).all():
        raise AssertionError('Observation end must be within the valid loan horizon.')
    train = out[out.split == 'train']
    # Descriptive historical benchmark, not a horizon-adjusted PD.
    by_grade = train.groupby('sub_grade')['default'].mean()
    out['benchmark_pd'] = out.sub_grade.map(by_grade).fillna(train.default.mean())
    out['pd_12m_proxy'] = (out.is_event.eq(1) & out.event_month.le(12)).astype(int)
    out['pd_24m_proxy'] = (out.is_event.eq(1) & out.event_month.le(24)).astype(int)
    pd_values = out[['pd_12m', 'pd_lifetime']].to_numpy()
    if not np.isfinite(pd_values).all():
        raise AssertionError('PD estimates must be finite.')
    if not (out.pd_12m.ge(0) & out.pd_12m.le(out.pd_lifetime) &
            out.pd_lifetime.le(1)).all():
        raise AssertionError('PDs must satisfy 0 <= pd_12m <= pd_lifetime <= 1.')
    evaluate_horizon(out, 12, 'pd_12m')
    report_36m_calibration(out)
    # Contractual lifetime horizon is term-specific; validate separately.
    for term in TERMS:
        evaluate_horizon(out[out.term_months == term], term, 'pd_lifetime')
    hazard_curves = make_hazard_curves(meta, model_curves)
    output_cols = ['id', 'issue_year', 'split', 'default', 'loan_status',
                   'term_months', 'sub_grade', 'months_to_last_pymnt',
                   'pd_lifetime', 'pd_12m', 'pd_12m_proxy', 'pd_24m_proxy',
                   'benchmark_pd', 'obs_end', 'is_event', 'event_month']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[output_cols].to_parquet(OUT, index=False)
    hazard_curves.to_parquet(HAZARD_CURVES_OUT, index=False)
    print(f'\nSaved {len(out):,} loans to {OUT}')
    print(f'Saved test-sample hazard curves to {HAZARD_CURVES_OUT}')
    print(out[output_cols].head().to_string(index=False))


if __name__ == '__main__':
    main()
