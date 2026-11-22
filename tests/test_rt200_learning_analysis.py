"""Verify RL gradients/history and strategy pairs after trial exclusions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/learning_rates'))
from analyze_rt200_learning import rl_objective,behavior,model_blocks


def test_rl_gradient_and_agreement_with_existing_likelihood():
    choices=np.array([1,0,1,1,0,1]);rewards=np.array([1.,0.,1.,0.,0.,1.])
    par=np.array([.35,.17,3.1]);mask=np.ones(6,bool)
    value,grad=rl_objective(par,[(choices,rewards,mask)])
    # Independently worked reward-value gaps before each of these six choices.
    gaps=np.array([0.,.175,.26,.37375,.2396625,.3102125])
    expected=np.sum(np.logaddexp(0.,par[2]*gaps)-choices*par[2]*gaps)
    assert np.isclose(value,expected)
    for used in [mask,np.array([1,0,1,0,1,1],bool)]:
        val,g=rl_objective(par,[(choices,rewards,used)])
        for j in range(3):
            step=np.eye(3)[j]*1e-6
            numeric=(rl_objective(par+step,[(choices,rewards,used)])[0]-rl_objective(par-step,[(choices,rewards,used)])[0])/2e-6
            assert np.isclose(g[j],numeric,atol=1e-7)


def test_omitted_likelihood_does_not_remove_feedback_history():
    par=np.array([.5,.2,2.])
    full=(np.array([1,1]),np.array([1.,1.]),np.array([False,True]))
    lost=(np.array([1]),np.array([1.]),np.array([True]))
    assert rl_objective(par,[full])[0] < rl_objective(par,[lost])[0]


def test_strategy_pairs_do_not_bridge_a_removed_trial_or_blocks():
    d=pd.DataFrame(dict(blockNumber=np.repeat([1,2],18),trialNumber=np.tile(np.arange(1,19),2),
        chosen=['face']*36,winner=['face']*36,feedback=['gain']*36,response_time=[500.]*36))
    d.loc[4,'response_time']=100.
    result,a=behavior(d,True)
    # Two blocks have 34 potential adjacent pairs; removing one interior trial loses two.
    assert result['n_win_pairs']==32
    assert result['winstay']==1 and result['n_retained']==35
    blocks=model_blocks(d,True)
    assert sum(len(x[0]) for x in blocks)==36
    assert sum(x[2].sum() for x in blocks)==35
