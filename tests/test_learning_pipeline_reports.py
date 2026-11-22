"""Check correction helpers and that current outputs use only eligible pairs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/learning_rates'))
from learning_analysis_helpers import adjust,spearman_resampling


def test_multiplicity_adjustment_and_missing_p_values():
    p=np.array([.01,.04,.03,np.nan])
    assert np.allclose(adjust(p)[:3],[.03,.04,.04])
    assert np.allclose(adjust(p,'holm')[:3],[.03,.06,.06])
    assert np.isnan(adjust(p)[3])


def test_correlation_reproducibility():
    d=pd.DataFrame({'x':np.arange(8),'y':[1,3,2,4,8,7,5,6]})
    a=spearman_resampling(d,'unit-check',199,100)
    b=spearman_resampling(d,'unit-check',199,100)
    assert a==b and a['n']==8 and .0<a['p_permutation']<=1


def test_analysis_outputs_never_include_excluded_or_incomplete_pairs():
    out=ROOT/'outputs/learning_rates/rt200_learning_analysis'
    if not (out/'all_correlations.csv').exists():
        raise AssertionError('Run the pipeline first; outputs are required for this integrity check')
    e=pd.read_csv(out/'participant_eligibility.csv')
    ids=set(e.loc[e.eligible_for_paired_analysis,'subject_id'].astype(int))
    for mode in ['rt200_primary','all_completed_sensitivity']:
        d=pd.read_csv(out/f'subject_measures_{mode}.csv')
        assert set(d.subject_id)==ids
    c=pd.read_csv(out/'all_correlations.csv')
    for r in c.itertuples():
        selected={int(i) for i in str(r.participant_ids).split(',') if i and i!='nan'}
        assert selected<=ids and len(selected)==r.n
    t=pd.read_csv(out/'retained_behavior_trials.csv')
    assert set(t.subject_id)==ids
    assert (t.loc[t['mode']=='rt200_primary','response_time']>=200).all()
    assert not t.duplicated(['subject_id','session','mode','blockNumber','trialNumber']).any()


def test_reports_are_present_and_include_exact_rules():
    out=ROOT/'outputs/learning_rates/task_engagement_screen_rt200'
    for filename in ['exclusion_criteria.txt','excluded_subjects_report.txt','engagement_screen_report.txt','included_subjects.txt']:
        assert (out/filename).stat().st_size>0
    text=(out/'exclusion_criteria.txt').read_text()
    assert '37 or more' in text and '46 or more' in text and 'dominant-key frequency' in text
    summary=ROOT/'outputs/learning_rates/rt200_learning_analysis/learning_analysis_summary.txt'
    assert summary.stat().st_size>0 and 'Holm across nine' in summary.read_text()
