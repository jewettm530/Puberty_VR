"""Shared statistical helpers for the task-engagement learning pipeline.

Stable per-comparison seeds, two-sided permutation Spearman tests, percentile
participant bootstrap intervals, and BH/Holm multiplicity corrections.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

METRICS=['alpha_plus','alpha_minus','beta','k','slope','accuracy','winstay','loseswitch','trial_gain']

LABEL={'alpha_plus':'Positive-feedback rate α+','alpha_minus':'Negative-feedback rate α−',
       'beta':'Choice consistency β','k':'Exponential curve rate k','slope':'Linear slope',
       'accuracy':'Optimal choices','winstay':'Win-stay','loseswitch':'Lose-switch','trial_gain':'Trial 18 − trial 1'}

def adjust(p,method='fdr'):
    p=np.asarray(p,float);q=np.full(len(p),np.nan);ix=np.flatnonzero(np.isfinite(p));ix=ix[np.argsort(p[ix])]
    if len(ix):
        if method=='holm':v=np.maximum.accumulate(p[ix]*np.arange(len(ix),0,-1))
        else:v=np.minimum.accumulate((p[ix]*len(ix)/np.arange(1,len(ix)+1))[::-1])[::-1]
        q[ix]=np.minimum(v,1)
    return q

def spearman_resampling(a,key,nperm=19999,nboot=3000):
    a=a.replace([np.inf,-np.inf],np.nan).dropna()
    n=len(a)
    result=dict(n=n,rho=np.nan,ci_low=np.nan,ci_high=np.nan,p_permutation=np.nan,p_asymptotic=np.nan)
    if n<6 or a.nunique().min()<2:return result
    xy=a.to_numpy(float)
    ranks=stats.rankdata(xy,axis=0);ranks-=ranks.mean(axis=0)
    x=ranks[:,0]/np.linalg.norm(ranks[:,0]);y=ranks[:,1]/np.linalg.norm(ranks[:,1])
    rho=float(x@y)
    # Stable per-comparison seed, independent of loop order and filtering.
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    indices=np.argsort(rng.random((nperm,n)),axis=1)
    null=y[indices]@x
    p=(1+np.sum(np.abs(null)>=abs(rho)-1e-12))/(nperm+1)
    samples=xy[rng.integers(0,n,(nboot,n))]
    boot=stats.rankdata(samples,axis=1);boot-=boot.mean(axis=1,keepdims=True)
    denom=np.linalg.norm(boot,axis=1).prod(axis=1)
    good=denom>0
    values=(boot[:,:,0]*boot[:,:,1]).sum(axis=1)[good]/denom[good]
    low,high=np.quantile(values,[.025,.975])
    result.update(rho=rho,ci_low=low,ci_high=high,p_permutation=p,
                  p_asymptotic=stats.spearmanr(xy[:,0],xy[:,1]).pvalue)
    return result
