"""Check strict screening thresholds and absolute lower-bound RT trimming."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1]/'scripts/learning_rates/screen_task_engagement.py'
spec = importlib.util.spec_from_file_location('task_screen', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frame(n=100):
    return pd.DataFrame(dict(blockNumber=np.ones(n), trialNumber=np.arange(n),
        chosen=['face']*n, response_choice_p=['left','right']*(n//2),
        response_time=np.tile([400.,600.],n//2)))


def test_missed_threshold_is_strict():
    d=frame(); d.loc[:19,['chosen','response_choice_p']]='none'
    row,_=module.screen_session(d)
    assert row['missed_fraction']==.20 and not row['flag_missed_gt20']
    d.loc[20,['chosen','response_choice_p']]='none'
    row,_=module.screen_session(d)
    assert row['flag_missed_gt20']


def test_dominant_key_thresholds_are_strict():
    d=frame(); d['response_choice_p']=['left']*90+['right']*10
    row,_=module.screen_session(d)
    assert not row['flag_key_gt90']
    d.loc[90,'response_choice_p']='left'
    row,_=module.screen_session(d)
    assert row['flag_key_gt90'] and not row['flag_key_gt95']
    d['response_choice_p']=['left']*96+['right']*4
    assert module.screen_session(d)[0]['flag_key_gt95']


def test_rt_screen_excludes_timeouts_and_has_no_upper_cutoff():
    d=frame(); d.loc[0,'response_time']=5000
    d.loc[1,['chosen','response_choice_p']]='none'; d.loc[1,'response_time']=999999
    d.loc[2,'response_time']=199
    row,t=module.screen_session(d)
    reference=d.loc[d.chosen!='none','response_time']
    assert row['median_rt_ms']==reference.median()
    assert np.isclose(row['sample_sd_rt_ms'],reference.std(ddof=1))
    assert t.rt_below_lower_bound.sum()==1
    assert t.loc[0,'retained_trial']
    assert row['rt_outlier_fraction_scheduled']==.01
    assert row['n_total_unusable']==2 and row['n_retained']==98


def test_stable_very_fast_responses_are_excluded():
    d=frame(); d['response_time']=50.
    row,t=module.screen_session(d)
    assert row['n_rt_outliers']==100 and not row['behavioral_pass']


def test_rt_and_proportion_boundaries_are_strict():
    d=frame(); d.loc[:24,'response_time']=199
    d.loc[25,'response_time']=200
    row,t=module.screen_session(d)
    assert row['rt_outlier_fraction_scheduled']==.25
    assert not row['flag_rt_removed_gt25'] and t.loc[25,'retained_trial']
    d.loc[25,'response_time']=199
    assert module.screen_session(d)[0]['flag_rt_removed_gt25']


def test_omissions_are_not_counted_as_fast_responses():
    d=frame(); d.loc[:19,['chosen','response_choice_p']]='none'
    d.loc[:19,'response_time']=0
    d.loc[20:44,'response_time']=100
    row,_=module.screen_session(d)
    assert row['n_rt_outliers']==25 and row['rt_outlier_fraction_scheduled']==.25
    assert row['total_unusable_fraction']==.45 and not row['flag_rt_removed_gt25']
