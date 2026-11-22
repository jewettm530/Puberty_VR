"""Generate readable reports from the current learning-analysis CSV outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from learning_analysis_helpers import METRICS, LABEL


def number(v, digits=4):
    return f'{v:.{digits}g}' if np.isfinite(v) else 'NA'


def correlation_line(r):
    return (f'{r.x} vs {r.y}: n={int(r.n)}, rho={number(r.rho)}, '
            f'95% CI [{number(r.ci_low)}, {number(r.ci_high)}], '
            f'unadjusted permutation p={number(r.p_permutation)}, BH q={number(r.q_domain)}')


def write_learning_reports(out):
    out=Path(out)
    paired=pd.read_csv(out/'paired_changes.csv')
    corr=pd.read_csv(out/'all_correlations.csv')
    eligibility=pd.read_csv(out/'participant_eligibility.csv')
    sessions=pd.read_csv(out/'recomputed_session_measures.csv')
    audit=pd.read_csv(out/'outlier_audit.csv')
    ids=eligibility.loc[eligibility.eligible_for_paired_analysis,'subject_id'].astype(int)
    current=sessions[(sessions['mode']=='rt200_primary') & sessions.paired_cohort]
    primary=paired[(paired['mode']=='rt200_primary') & (paired.screen=='exclude_2sd')]
    c=corr[(corr['mode']=='rt200_primary') & (corr.screen=='exclude_2sd')]
    core=c[c.domain=='core_learning']
    substantive=core[(core.family!='level_vs_change_coupled') & (core.q_domain<.05)]
    hr=c[c.domain=='supplemental_hr']
    lines=[f'LEARNING ANALYSIS SUMMARY: {len(ids)} INCLUDED PAIRED PARTICIPANTS',
        'Included IDs: '+', '.join(f'{i:03d}' for i in ids),
        f'Primary analysis: {int(current.n_retained.sum())} retained choices across {len(current)} sessions. Missed choices and RT <200 ms are omitted from behavioral summaries.',
        'Participants are selected only from the current paired eligibility file. Screen-eligible single sessions additionally contribute to level-specific outlier references; they do not enter paired comparisons or correlations.',
        'Outlier screening: one-pass |z|>=2, separately by measure. A participant is omitted only from affected comparisons. For change correlations, flags on the change score and its pre/post components are considered. Model-boundary exclusions are a separate sensitivity analysis.',
        'RL history: fast responses are not scored in choice likelihood, but their observed choices and feedback still update Q. Missing choices do not update Q. Values reset to 0.5 at each block. Strategy pairs do not bridge omissions, fast trials, or block boundaries.',
        'STATISTICAL PROCEDURES',
        'Average changes: paired Wilcoxon for alpha+, alpha-, beta and k; paired t tests for other indices. Holm corrections across all nine core outcomes and within smaller families are both reported. Mean-change CIs are unadjusted paired t intervals.',
        'Correlations: two-sided Monte Carlo permutation Spearman tests and participant bootstrap 95% CIs. BH-FDR across 351 core pairs; 279 additional RPE-related pairs are secondary. A broader 630-learning-test correction is also exported. HR, when available/enabled, has a separate 189-test correction. q<.05 denotes FDR significance; raw p<.05 alone does not.',
        '\nAVERAGE PRE-TO-POST CHANGES (MEASURE-SPECIFIC 2 SD FILTER)']
    for r in primary.itertuples():
        scale=100 if r.measure in ['accuracy','winstay','loseswitch','trial_gain','slope'] else 1
        unit=' percentage points/trial' if r.measure=='slope' else ' percent / percentage-point change' if scale==100 else ''
        lines.append(f'{LABEL.get(r.measure,r.measure)}{unit}: n={int(r.n)}; '
            f'pre mean={number(scale*r.mean_pre)}, SD={number(scale*r.sd_pre)}; '
            f'post mean={number(scale*r.mean_post)}, SD={number(scale*r.sd_post)}; '
            f'mean change={number(scale*r.change)}, 95% CI [{number(scale*r.ci_low)}, {number(scale*r.ci_high)}]; '
            f'raw p={number(r.p_primary)}, Holm across nine={number(r.p_holm_core9)}, within-family Holm={number(r.p_holm_family)}. '
            f'Increased/decreased/unchanged: {r.increased}/{r.decreased}/{r.unchanged}.')
    sig9=primary[primary.measure.isin(METRICS) & (primary.p_holm_core9<.05)]
    lines.append('Significant average changes across all nine core outcomes: '+(', '.join(sig9.measure) or 'none')+'.')
    for m in ['winstay','loseswitch']:
        r=primary[primary.measure==m].iloc[0]
        lines.append(f'{m}: {"significant" if r.p_holm_family<.05 else "not significant"} across two strategy tests '
            f'(adjusted p={number(r.p_holm_family)}); '
            f'{"significant" if r.p_holm_core9<.05 else "not significant"} across nine outcomes (adjusted p={number(r.p_holm_core9)}).')
    lines.append('\nSIGNIFICANT CORE CORRELATIONS WITHOUT LEVEL/CHANGE SHARED-COMPONENT PAIRS')
    if substantive.empty:lines.append('None survived BH-FDR across all 351 core pairs.')
    for r in substantive.sort_values(['family','q_domain','x','y']).itertuples():
        lines.append(f'[{r.family}] '+correlation_line(r))
    lines.extend(['Interpretation: positive correlations indicate that higher values accompany higher values; negative correlations indicate an inverse association. Win-stay/accuracy associations describe rewarded-choice repetition accompanying optimal-choice performance, not proof that strategy caused improvement.',
        'Changes in accuracy/slope/k/alpha+ associated with strategy changes describe differences among individuals even if the average pre/post change is nonsignificant. They do not establish prediction in new participants.',
        '\nSENSITIVITY OF KEY STRATEGY ASSOCIATIONS'])
    specs=[('accuracy_pre','winstay_pre'),('accuracy_post','winstay_post'),
        ('slope_pre','winstay_pre'),('slope_post','winstay_post'),
        ('delta_accuracy','delta_winstay'),('delta_slope','delta_winstay'),('delta_k','delta_winstay'),
        ('delta_alpha_plus','delta_winstay'),('slope_post','loseswitch_post'),('delta_accuracy','delta_loseswitch')]
    for x,y in specs:
        rows=corr[((corr.x==x)&(corr.y==y))|((corr.x==y)&(corr.y==x))]
        lines.append(f'{x} vs {y}:')
        for r in rows.itertuples():
            lines.append(f'  {r.mode}, {r.screen}: n={int(r.n)}, rho={number(r.rho)}, p={number(r.p_permutation)}, q={number(r.q_domain)}.')
        lines.append('  '+('Survives correction in all four analyses.' if len(rows)==4 and (rows.q_domain<.05).all() else 'Sensitivity-dependent or not consistently significant; avoid treating as a uniformly robust finding.'))
    boundary_path=out/'significant_associations_boundary_sensitivity.csv'
    try:boundary=pd.read_csv(boundary_path)
    except pd.errors.EmptyDataError:boundary=pd.DataFrame()
    lines.append('\nFIT-BOUNDARY SENSITIVITY')
    lines.append(f'{int(current.any_boundary.sum())}/{len(current)} paired-cohort session fits reached at least one RL bound; '
        f'alpha+ {int(current.alpha_plus_boundary.sum())}, alpha- {int(current.alpha_minus_boundary.sum())}, beta {int(current.beta_boundary.sum())}; '
        f'k at a bound: {int(current.k_boundary.sum())}. Optimizer failures: {int((~current.optimizer_success).sum())}.')
    lines.append('Boundary-free correlation rechecks are sensitivity results selected after the primary analysis, not independent confirmation:')
    for r in boundary.itertuples():
        lines.append(f'  {r.x} vs {r.y}: n={int(r.n)}, rho={number(r.rho)}, raw p={number(r.p_permutation)}.')
    lines.append('\nWITHIN-SESSION ACQUISITION AND FIRST-FIVE/LAST-FIVE COMPARISONS')
    for filename in ['learning_acquisition_tests.csv','first5_last5_tests.csv']:
        a=pd.read_csv(out/filename);a=a[a['mode']=='rt200_primary']
        for r in a.itertuples():
            lines.append(f'{filename}, session {r.session}, {r.measure}: n={int(r.n)}, '
                f'first={number(r.mean_pre)}, last={number(r.mean_post)}, change={number(r.change)}, '
                f'raw p={number(r.p_primary)}, Holm p={number(r.p_holm)}.')
    lines.append('\nSAME-INDEX PRE/POST ASSOCIATIONS')
    for m in METRICS:
        r=core[(core.x==m+'_pre') & (core.y==m+'_post')].iloc[0]
        lines.append(correlation_line(r))
    lines.append('\nEXPLORATORY MODEL-DERIVED RPE INDICES')
    rpe=c[c.domain=='exploratory_rpe']
    lines.append(f'{int((rpe.q_domain<.05).sum())} of {len(rpe)} RPE-related correlations survived the separate FDR correction. '
        'RPE/expected-value quantities are algebraically related to fitted model parameters and rewards. Their associations can follow from model equations and should not be counted as independent mechanistic discoveries. See exploratory_rpe_correlations.csv.')
    lines.append('\nSUPPLEMENTAL HR')
    if hr.empty:lines.append('Not run: disabled or processed HR input unavailable. Core learning results do not require HR.')
    else:
        lines.append(f'{int((hr.q_domain<.05).sum())}/{len(hr)} HR/learning associations survived separate BH correction.')
        for r in hr.sort_values('p_permutation').head(10).itertuples():lines.append(correlation_line(r))
    lines.append('\nMEASURE-SPECIFIC OUTLIERS IN THE PAIRED COHORT')
    a=audit[(audit['mode']=='rt200_primary') & audit.paired_cohort & audit.flagged_2sd]
    for measure,b in a.groupby('measure'):
        lines.append(measure+': '+', '.join(f'{int(i):03d}' for i in sorted(b.subject_id))+'.')
    lines.extend(['\nLIMITATIONS AND INTERPRETATION',
        'Retrospective engagement screening does not prove motivation. No control condition isolates VR stress from practice, time, or order effects. Non-significance does not establish no effect.',
        'Accuracy, slope, strategies, and fitted parameters share choice data. Level/change correlations additionally share mathematical components; see their explicit family label in the CSV. CIs are not multiplicity-adjusted.',
        'All tested correlations, not just significant ones, are in all_correlations.csv, with participant IDs, sample sizes, raw p-values, corrected q-values, and bootstrap intervals.',
        'All PNGs are generated by analyze_rt200_learning.py. The detailed machine-readable audit is learning_analysis_report.txt; full parameters, SDs, and individual directions are in the CSVs.'])
    text='\n\n'.join(lines)+'\n'
    (out/'learning_analysis_summary.txt').write_text(text)
    (out/'READ_ME_results_and_interpretation.md').write_text(text)
