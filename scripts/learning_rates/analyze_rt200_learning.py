#!/usr/bin/env python3
"""Recompute learning in the RT-screened paired cohort and explore associations.

Primary: completed responses >=200 ms. RL choice likelihood omits faster
responses, but their observed choice/feedback still updates Q; block resets
and original trial positions are preserved. Strategy pairs must be consecutive
original trials in a block, both retained. Sensitivity retains all completed
choices in the SAME screened cohort. No claim of causal stress effects.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from learning_analysis_helpers import adjust, spearman_resampling, METRICS, LABEL

RPE=['mean_absolute_rpe','mean_positive_rpe','mean_expected_reward']
ALL=METRICS+RPE
BOUNDS=[(.001,.999),(.001,.999),(.01,20.)]
STARTS=[(.2,.2,2.),(.5,.5,5.),(.8,.2,5.),(.2,.8,5.),(.8,.8,10.)]


def rl_objective(par, blocks):
    """NLL and analytic gradient; every valid outcome updates latent values."""
    ap,am,beta=par; total=0.; gradient=np.zeros(3)
    for choices,rewards,mask in blocks:
        q=np.array([.5,.5]); dq=np.zeros((2,3))
        for c,r,keep in zip(choices,rewards,mask):
            gap=q[1]-q[0]; z=beta*gap
            if keep:
                total+=np.logaddexp(0.,z)-c*z
                dz=beta*(dq[1]-dq[0]); dz[2]+=gap
                gradient+=(expit(z)-c)*dz
            error=r-q[c]; index=0 if error>=0 else 1
            rate=ap if index==0 else am
            dq[c]=(1-rate)*dq[c]; dq[c,index]+=error
            q[c]+=rate*error
    return float(total),gradient


def model_blocks(d,trim):
    blocks=[]
    for _,b in d.groupby('blockNumber',sort=True):
        b=b[b.chosen.isin(['face','house']) & b.feedback.isin(['gain','loss'])]
        blocks.append(((b.chosen=='face').to_numpy(int),(b.feedback=='gain').to_numpy(float),
                       (b.response_time>=200).to_numpy() if trim else np.ones(len(b),bool)))
    return blocks


def fit_rl(d,trim):
    blocks=model_blocks(d,trim)
    fits=[minimize(rl_objective,start,args=(blocks,),jac=True,method='L-BFGS-B',bounds=BOUNDS,
                   options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7}) for start in STARTS]
    converged=[r for r in fits if r.success and np.isfinite(r.fun)]
    best=min(converged or fits,key=lambda r:r.fun)
    ap,am,beta=best.x
    result=dict(alpha_plus=ap,alpha_minus=am,beta=beta,NLL=best.fun,convergence=best.status,
                optimizer_success=best.success,n_likelihood_trials=sum(int(b[2].sum()) for b in blocks))
    for name,v,(low,high) in zip(['alpha_plus','alpha_minus','beta'],best.x,BOUNDS):
        tolerance=.0001 if name!='beta' else .01
        result[f'{name}_boundary']=bool(v<=low+tolerance or v>=high-tolerance)
    result['any_boundary']=any(result[f'{m}_boundary'] for m in ['alpha_plus','alpha_minus','beta'])
    states=[]
    for block,(_,b) in zip(blocks,d.groupby('blockNumber',sort=True)):
        b=b[b.chosen.isin(['face','house']) & b.feedback.isin(['gain','loss'])]
        q=np.array([.5,.5])
        for trial,c,r,keep in zip(b.itertuples(),*block):
            error=r-q[c]
            states.append(dict(blockNumber=trial.blockNumber,trialNumber=trial.trialNumber,
                chosen=trial.chosen,feedback=trial.feedback,rt_ms=trial.response_time,
                included_likelihood=keep,expected_reward=q[c],rpe=error))
            q[c]+=(ap if error>=0 else am)*error
    states=pd.DataFrame(states); use=states[states.included_likelihood]
    result.update(mean_absolute_rpe=use.rpe.abs().mean(),
        mean_positive_rpe=use.loc[use.rpe>0,'rpe'].mean(),mean_expected_reward=use.expected_reward.mean())
    return result,states


def curve_fit_values(d):
    p=d.groupby('trialNumber').optimal.mean()
    x=p.index.to_numpy(float);y=p.to_numpy(float)
    slope=np.polyfit(x,y,1)[0] if len(x)>1 else np.nan
    if len(x)<4:
        return dict(slope=slope,k=np.nan,fit_a=np.nan,fit_b=np.nan,curve_fit_status='insufficient_positions',
            k_boundary=False,curve_any_boundary=False,accuracy=d.optimal.mean(),
            trial_gain=p.get(18,np.nan)-p.get(1,np.nan),trial1=p.get(1,np.nan),trial18=p.get(18,np.nan))
    a0=min(max(y.max(),.51),1); b0=max(a0-y.min(),.01)
    try:
        fit,_=curve_fit(lambda t,a,b,k:a-b*np.exp(-k*t),x,y,p0=[a0,b0,.05],
                        bounds=([0,0,0],[1,1,2]),maxfev=10000)
        a,b,k=fit; status='success'
    except (ValueError,RuntimeError):a=b=k=np.nan;status='failed'
    return dict(slope=slope,k=k,fit_a=a,fit_b=b,curve_fit_status=status,
                k_boundary=bool(np.isfinite(k) and (k<.0001 or k>1.9999)),
                curve_any_boundary=bool(np.isfinite(k) and (a<.0001 or a>.9999 or b<.0001 or b>.9999 or k<.0001 or k>1.9999)),
                accuracy=d.optimal.mean(),trial_gain=p.get(18,np.nan)-p.get(1,np.nan),
                trial1=p.get(1,np.nan),trial18=p.get(18,np.nan))


def behavior(d,trim):
    d=d.sort_values(['blockNumber','trialNumber']).copy()
    d['valid']=d.chosen.isin(['face','house'])
    d['keep']=d.valid & ((d.response_time>=200) if trim else True)
    d['optimal']=(d.chosen==d.winner).astype(float)
    a=d[d.keep]; result=curve_fit_values(a)
    adjacent=d.blockNumber.eq(d.blockNumber.shift()) & d.trialNumber.eq(d.trialNumber.shift()+1)
    both=d.keep & d.keep.shift(fill_value=False) & adjacent
    win=both & d.feedback.shift().eq('gain');lose=both & d.feedback.shift().eq('loss')
    stay=d.chosen.eq(d.chosen.shift())
    result.update(winstay=float(stay[win].mean()),loseswitch=float((~stay)[lose].mean()),
                  n_win_pairs=int(win.sum()),n_loss_pairs=int(lose.sum()),
                  mean_rt_ms=a.response_time.mean(),median_rt_ms=a.response_time.median(),n_retained=len(a))
    for half,subset in [('first5',a[a.blockNumber<=5]),('last5',a[a.blockNumber>5])]:
        result[f'accuracy_{half}']=subset.optimal.mean()
        result[f'mean_rt_ms_{half}']=subset.response_time.mean()
        result[f'slope_{half}']=curve_fit_values(subset)['slope']
    return result,a


def paired_stats(a,b,measure):
    use=pd.concat([a,b],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    pre,post=use.iloc[:,0],use.iloc[:,1];delta=post-pre;n=len(delta)
    row=dict(measure=measure,n=n,mean_pre=pre.mean(),sd_pre=pre.std(ddof=1),mean_post=post.mean(),
             sd_post=post.std(ddof=1),change=delta.mean(),sd_change=delta.std(ddof=1),
             increased=int((delta>1e-12).sum()),decreased=int((delta< -1e-12).sum()),
             unchanged=int((delta.abs()<=1e-12).sum()),participant_ids=','.join(f'{int(i):03d}' for i in use.index))
    if n<3:return row|dict(p_t=np.nan,p_w=np.nan,p_primary=np.nan,ci_low=np.nan,ci_high=np.nan,dz=np.nan)
    sd=delta.std(ddof=1);se=sd/np.sqrt(n)
    ci=stats.t.interval(.95,n-1,loc=delta.mean(),scale=se) if se>0 else (delta.mean(),delta.mean())
    p_t=stats.ttest_rel(post,pre).pvalue if sd>0 else (1. if delta.mean()==0 else 0.)
    p_w=stats.wilcoxon(delta).pvalue if (delta!=0).any() else 1.
    return row|dict(p_t=p_t,p_w=p_w,p_primary=p_w if measure in ['alpha_plus','alpha_minus','beta','k'] else p_t,
                    ci_low=ci[0],ci_high=ci[1],dz=delta.mean()/sd if sd else np.nan)


def classify_pair(x,y):
    def kind(v):return 'delta' if v.startswith('delta_') else v.rsplit('_',1)[1]
    sx,sy=kind(x),kind(y)
    if sx==sy:return f'within_{sx}'
    if 'delta' in [sx,sy]:return 'level_vs_change_coupled'
    return 'pre_vs_post'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--permutations',type=int,default=19999)
    p.add_argument('--bootstraps',type=int,default=3000)
    p.add_argument('--skip-hr',action='store_true',help='Run learning analyses without supplemental HR')
    args=p.parse_args();root=args.project_root
    if args.permutations<1 or args.bootstraps<10:
        raise ValueError('Use at least 1 permutation and 10 bootstrap samples')
    out=root/'outputs/learning_rates/rt200_learning_analysis';out.mkdir(parents=True,exist_ok=True)
    eligibility=pd.read_csv(root/'outputs/learning_rates/task_engagement_screen_rt200/participant_eligibility.csv')
    eligible_sessions=pd.read_csv(root/'outputs/learning_rates/task_engagement_screen_rt200/session_engagement_screen.csv')
    ids=eligibility.loc[eligibility.eligible_for_paired_analysis,'subject_id'].astype(int).tolist()
    derived_ids={int(subject) for subject,a in eligible_sessions.groupby('subject_id')
                 if set(a.session)=={1,2} and a.analysis_eligible.all()}
    if set(ids)!=derived_ids or len(ids)!=len(set(ids)):
        raise ValueError('Participant eligibility does not match session-level screening; rerun the screen')
    if len(ids)<6:raise ValueError(f'At least six eligible complete pairs are required; found {len(ids)}')
    config_path=root/'outputs/learning_rates/task_engagement_screen_rt200/screen_configuration.json'
    if not config_path.exists():raise ValueError('Run screen_task_engagement.py first to generate its configuration and input audit')
    config=json.loads(config_path.read_text())
    if config['min_rt_ms']!=200:raise ValueError('This learning analysis requires the decided 200-ms screen')
    for record in eligible_sessions.itertuples():
        subject,session=int(record.subject_id),int(record.session)
        path=root/f'data/processed/plearning/{subject:03d}_plearning_{session}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['input_sha256'].get(f'{subject:03d}:{session}'):
            raise ValueError(f'Input changed since screening: {path}; rerun the screen')
    eligibility.to_csv(out/'participant_eligibility.csv',index=False)
    results=[];rpes=[];curves=[];retained_frames=[]
    for mode,trim in [('rt200_primary',True),('all_completed_sensitivity',False)]:
        # References include all screen-eligible sessions, including single-session subjects.
        for record in eligible_sessions[eligible_sessions.analysis_eligible].itertuples():
            subject=int(record.subject_id);session=int(record.session)
            path=root/f'data/processed/plearning/{subject:03d}_plearning_{session}.csv'
            d=pd.read_csv(path).sort_values(['blockNumber','trialNumber'])
            result,a=behavior(d,trim);rl,state=fit_rl(d,trim)
            result.update(rl);result.update(subject_id=subject,session=session,mode=mode,paired_cohort=subject in ids)
            results.append(result)
            state['subject_id']=subject;state['session']=session;state['mode']=mode;rpes.append(state)
            if subject in ids:
                for trial,b in a.groupby('trialNumber'):
                    curves.append(dict(subject_id=subject,session=session,mode=mode,trial=trial,
                                       optimal=b.optimal.mean(),n_trials=len(b)))
                a=a.copy();a['subject_id']=subject;a['session']=session;a['mode']=mode;retained_frames.append(a)
        print(f'Completed {mode} refits.',flush=True)
    long=pd.DataFrame(results);long.to_csv(out/'recomputed_session_measures.csv',index=False)
    pd.concat(rpes).to_csv(out/'refitted_trial_rpes_with_history.csv',index=False)
    pd.concat(retained_frames).to_csv(out/'retained_behavior_trials.csv',index=False)
    curves=pd.DataFrame(curves);curves.to_csv(out/'individual_trial_position_curves.csv',index=False)
    hr_path=root/'data/analysis_data/heart_rate/heart_rate_summary_wide.csv'
    use_hr=not args.skip_hr and hr_path.exists()
    h=pd.read_csv(hr_path).set_index('subject_id') if use_hr else pd.DataFrame(columns=[
        'vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr','pre_vr_mean_hr',
        'prep_hr_reactivity','speech_hr_reactivity','math_hr_reactivity'],dtype=float)
    if not use_hr:
        print('Supplemental HR skipped (disabled or input unavailable).',flush=True)
        (out/'supplemental_hr_correlations.csv').unlink(missing_ok=True)
    # HR remains supplemental; equal-weight speech/math mean requires both stages.
    h['speech_math_hr_reactivity']=h[['vr_speech_mean_hr','vr_math_mean_hr']].mean(axis=1,skipna=False)-h.pre_vr_mean_hr
    h['all_stage_mean_hr_reactivity']=h[['vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr']].mean(axis=1,skipna=False)-h.pre_vr_mean_hr
    h['max_stage_hr_reactivity']=h[['vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr']].max(axis=1)-h.pre_vr_mean_hr
    h['absolute_mean_vr_hr']=h[['vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr']].mean(axis=1,skipna=False)
    hrs=['speech_math_hr_reactivity','prep_hr_reactivity','speech_hr_reactivity','math_hr_reactivity',
         'all_stage_mean_hr_reactivity','max_stage_hr_reactivity','absolute_mean_vr_hr']
    if not use_hr:hrs=[]
    wide_by_mode={};flags_by_mode={};boundaries={};audit=[];paired=[];records=[];acquisitions=[];half_tests=[]
    hr_components={'speech_math_hr_reactivity':['pre_vr_mean_hr','vr_speech_mean_hr','vr_math_mean_hr'],
        'prep_hr_reactivity':['pre_vr_mean_hr','vr_prep_mean_hr'],
        'speech_hr_reactivity':['pre_vr_mean_hr','vr_speech_mean_hr'],
        'math_hr_reactivity':['pre_vr_mean_hr','vr_math_mean_hr'],
        'all_stage_mean_hr_reactivity':['pre_vr_mean_hr','vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr'],
        'max_stage_hr_reactivity':['pre_vr_mean_hr','vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr'],
        'absolute_mean_vr_hr':['vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr']}
    for mode,reference in long.groupby('mode',sort=False):
        wide=pd.DataFrame(index=sorted(reference.subject_id.unique()))
        for m in ALL+['trial1','trial18']:
            for session,tag in [(1,'pre'),(2,'post')]:
                wide[f'{m}_{tag}']=reference[reference.session==session].set_index('subject_id')[m]
            wide[f'delta_{m}']=wide[f'{m}_post']-wide[f'{m}_pre']
        if use_hr:wide=wide.join(h[hrs+['pre_vr_mean_hr','vr_prep_mean_hr','vr_speech_mean_hr','vr_math_mean_hr']])
        flag={};boundary={}
        for col in wide:
            # Level references: all eligible sessions. Change reference: paired included subjects.
            ref=wide.loc[ids,col].dropna() if col.startswith('delta_') else wide[col].dropna()
            mean,sd=ref.mean(),ref.std(ddof=1)
            z=(wide[col]-mean)/sd if sd>0 else wide[col]*0
            flag[col]=set(z.index[z.abs()>=2])
            for subject in wide.index[wide[col].notna()]:
                audit.append(dict(mode=mode,subject_id=subject,measure=col,value=wide.loc[subject,col],
                    reference_n=len(ref),reference_mean=mean,reference_sd=sd,z=z.loc[subject],
                    flagged_2sd=subject in flag[col],paired_cohort=subject in ids))
        for col in [f'{m}_{tag}' for m in ['alpha_plus','alpha_minus','beta','k'] for tag in ['pre','post']]:
            m,tag=col.rsplit('_',1);ses=1 if tag=='pre' else 2
            b=reference[reference.session==ses]
            boundary[col]=set(b.loc[b[f'{m}_boundary'] | ~b.optimizer_success,'subject_id'])
        for m in ['alpha_plus','alpha_minus','beta','k']:
            boundary[f'delta_{m}']=boundary[f'{m}_pre']|boundary[f'{m}_post']
        d=wide.loc[ids].copy();d.index.name='subject_id';d.to_csv(out/f'subject_measures_{mode}.csv')
        wide_by_mode[mode]=d;flags_by_mode[mode]=flag;boundaries[mode]=boundary

        def select(cols,screen,extras=None):
            bad=set()
            expanded=list(cols)+list(extras or [])
            for c in cols:
                if c.startswith('delta_'):
                    expanded.extend([c[6:]+'_pre',c[6:]+'_post'])
                if c in hr_components:expanded.extend(hr_components[c])
            if screen!='all_values':
                for c in expanded:bad |= flag.get(c,set())
            if screen=='exclude_2sd_and_boundaries':
                for c in expanded:bad |= boundary.get(c,set())
            return d.loc[~d.index.isin(bad),cols].replace([np.inf,-np.inf],np.nan).dropna()

        screens=['all_values','exclude_2sd','exclude_2sd_and_boundaries']
        for screen in screens:
            for m in ALL:
                a=select([f'{m}_pre',f'{m}_post'],screen)
                r=paired_stats(a.iloc[:,0],a.iloc[:,1],m);r.update(mode=mode,screen=screen)
                paired.append(r)
        for session,tag in [(1,'pre'),(2,'post')]:
            c=curves[(curves['mode']==mode)&(curves.session==session)].pivot(index='subject_id',columns='trial',values='optimal')
            r=paired_stats(c[1],c[18],'trial18_vs_trial1');r.update(mode=mode,session=session);acquisitions.append(r)
            b=reference[(reference.session==session)&reference.subject_id.isin(ids)].set_index('subject_id')
            for m in ['accuracy','mean_rt_ms','slope']:
                r=paired_stats(b[f'{m}_first5'],b[f'{m}_last5'],m);r.update(mode=mode,session=session);half_tests.append(r)
        # All unique pairs among pre, post and change values. Level/change
        # correlations are explicitly labeled because they share components.
        columns=[f'{m}_{tag}' for tag in ['pre','post'] for m in ALL]+[f'delta_{m}' for m in ALL]
        for screen in ['all_values','exclude_2sd']:
            for x,y in combinations(columns,2):
                core=all((c[6:] if c.startswith('delta_') else c.rsplit('_',1)[0]) in METRICS for c in [x,y])
                a=select([x,y],screen)
                r=spearman_resampling(a,f'rt200|{mode}|{screen}|{x}|{y}',args.permutations,args.bootstraps)
                r.update(mode=mode,screen=screen,x=x,y=y,domain='core_learning' if core else 'exploratory_rpe',
                         family=classify_pair(x,y),participant_ids=','.join(f'{int(i):03d}' for i in a.index))
                records.append(r)
            # HR is secondary and is tested against the nine core indices.
            for hr in hrs:
                for y in [c for c in columns if (c[6:] if c.startswith('delta_') else c.rsplit('_',1)[0]) in METRICS]:
                    a=select([hr,y],screen)
                    r=spearman_resampling(a,f'rt200|{mode}|{screen}|{hr}|{y}',args.permutations,args.bootstraps)
                    r.update(mode=mode,screen=screen,x=hr,y=y,domain='supplemental_hr',family='hr_vs_learning',
                             participant_ids=','.join(f'{int(i):03d}' for i in a.index));records.append(r)
            print(f'Completed correlations: {mode}, {screen}.',flush=True)
    paired=pd.DataFrame(paired);paired['p_holm_core9']=np.nan;paired['p_holm_family']=np.nan
    for _,a in paired.groupby(['mode','screen']):
        core=a[a.measure.isin(METRICS)];paired.loc[core.index,'p_holm_core9']=adjust(core.p_primary,'holm')
        for family in [['alpha_plus','alpha_minus','beta'],['winstay','loseswitch'],['k','slope','accuracy','trial_gain'],RPE]:
            b=a[a.measure.isin(family)];paired.loc[b.index,'p_holm_family']=adjust(b.p_primary,'holm')
    paired.to_csv(out/'paired_changes.csv',index=False)
    for frame,name in [(pd.DataFrame(acquisitions),'learning_acquisition'),(pd.DataFrame(half_tests),'first5_last5')]:
        frame['p_holm']=np.nan
        for _,a in frame.groupby('mode'):frame.loc[a.index,'p_holm']=adjust(a.p_primary,'holm')
        frame.to_csv(out/f'{name}_tests.csv',index=False)
    audit=pd.DataFrame(audit);audit.to_csv(out/'outlier_audit.csv',index=False)
    audit[audit.flagged_2sd & audit.paired_cohort].to_csv(out/'paired_cohort_outliers.csv',index=False)
    corr=pd.DataFrame(records);corr['q_domain']=np.nan;corr['q_family']=np.nan;corr['q_all_learning630']=np.nan
    for _,a in corr.groupby(['mode','screen','domain']):corr.loc[a.index,'q_domain']=adjust(a.p_permutation)
    for _,a in corr.groupby(['mode','screen','domain','family']):corr.loc[a.index,'q_family']=adjust(a.p_permutation)
    for _,a in corr[corr.domain!='supplemental_hr'].groupby(['mode','screen']):corr.loc[a.index,'q_all_learning630']=adjust(a.p_permutation)
    corr.to_csv(out/'all_correlations.csv',index=False)
    for domain,a in corr.groupby('domain'):a.to_csv(out/f'{domain}_correlations.csv',index=False)
    # Boundary sensitivity for primary core associations already surviving FDR.
    b_records=[]
    for r in corr[(corr['mode']=='rt200_primary')&(corr.screen=='exclude_2sd')&(corr.domain=='core_learning')&(corr.q_domain<.05)].itertuples():
        d=wide_by_mode[r.mode];bad=set()
        for c in [r.x,r.y]:bad|=boundaries[r.mode].get(c,set())
        ids2=[int(i) for i in r.participant_ids.split(',') if int(i) not in bad]
        a=d.loc[ids2,[r.x,r.y]]
        b=spearman_resampling(a,f'boundary|{r.x}|{r.y}',args.permutations,args.bootstraps)
        b.update(x=r.x,y=r.y,primary_q=r.q_domain);b_records.append(b)
    pd.DataFrame(b_records).to_csv(out/'significant_associations_boundary_sensitivity.csv',index=False)
    make_figures(out,wide_by_mode['rt200_primary'],paired,corr,curves)
    manifest=dict(n_paired=len(ids),included_ids=[f'{i:03d}' for i in ids],permutations=args.permutations,
        bootstrap_samples=args.bootstraps,core_correlations_per_mode_screen=351,
        exploratory_rpe_correlations_per_mode_screen=279,supplemental_hr_correlations_per_mode_screen=189 if use_hr else 0,
        methods=__doc__,outlier_reference=f'all screen-eligible sessions for levels; {len(ids)} paired subjects for deltas; one-pass |z|>=2',
        corrections='BH-FDR across 351 core, 279 RPE-related, 189 HR tests separately per mode/screen; broader 630 learning correction also supplied',
        interval_note='Correlation CIs: paired percentile bootstrap, unadjusted. Mean-change CIs: paired t interval, not multiplicity adjusted.',
        primary_paired_tests='Wilcoxon for alpha+, alpha-, beta, k; paired t for remaining indices; Holm across 9 core and within families',
        rpe_note='Model-derived descriptive RPE quantities do not measure neural RPE sensitivity.',
        limitations=['Retrospective task-engagement criteria; no validated diagnosis of motivation.',
            'No control condition; session effects are not uniquely attributable to VR stress.',
            'Choice-based indices share data, so correlations may be partly mathematical.',
            'Level/change correlations contain shared components and regression-to-mean effects.',
            'Fast-trial Q updates assume the recorded feedback still contributes to learning; all-completed sensitivity supplied.',
            'Sparse retained trial positions can make k and strategy estimates unstable; boundary audit supplied.'])
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    primary=corr[(corr['mode']=='rt200_primary')&(corr.screen=='exclude_2sd')]
    lines=[f'RT200-SCREENED LEARNING ANALYSIS: {len(ids)} PAIRED PARTICIPANTS',json.dumps(manifest,indent=2),
        '\nPRIMARY PAIRED CHANGES (MEASURE-SPECIFIC 2 SD FILTER):',paired[(paired['mode']=='rt200_primary')&(paired.screen=='exclude_2sd')].to_string(index=False),
        '\nPRIMARY CORE CORRELATIONS WITH BH q<.05 ACROSS ALL 351:',primary[(primary.domain=='core_learning')&(primary.q_domain<.05)].sort_values('q_domain').to_string(index=False),
        '\nPRIMARY RPE-RELATED CORRELATIONS WITH q<.05 ACROSS 279:',primary[(primary.domain=='exploratory_rpe')&(primary.q_domain<.05)].sort_values('q_domain').to_string(index=False),
        '\nPRIMARY SUPPLEMENTAL HR CORRELATIONS WITH q<.05 ACROSS 189:',primary[(primary.domain=='supplemental_hr')&(primary.q_domain<.05)].sort_values('q_domain').to_string(index=False),
        '\nMODEL DIAGNOSTICS:',long.groupby('mode')[['optimizer_success','any_boundary','alpha_plus_boundary','alpha_minus_boundary','beta_boundary','k_boundary','curve_any_boundary']].sum().to_string()]
    (out/'learning_analysis_report.txt').write_text('\n'.join(lines)+'\n')
    write_readable_summary(out)
    print(f'Analysis complete for {len(ids)} pairs. Read {out / "learning_analysis_summary.txt"}',flush=True)


def make_figures(out,d,paired,corr,curves):
    primary=corr[(corr['mode']=='rt200_primary')&(corr.screen=='exclude_2sd')]
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for ax,m in zip(axes.flat,METRICS):
        r=paired[(paired['mode']=='rt200_primary')&(paired.screen=='exclude_2sd')&(paired.measure==m)].iloc[0]
        ids=[int(i) for i in r.participant_ids.split(',')];a=d.loc[ids,[m+'_pre',m+'_post']]
        scale=100 if m in ['slope','accuracy','winstay','loseswitch','trial_gain'] else 1
        for _,row in a.iterrows():ax.plot([1,2],row.to_numpy()*scale,color='#62828a',alpha=.4,lw=.8)
        ax.plot([1,2],a.mean().to_numpy()*scale,'o-',color='#b75b3c',lw=2.5)
        ax.set(xticks=[1,2],xticklabels=['Before VR','After VR'],title=f'{LABEL[m]} (n={len(a)})\nHolm across 9 p={r.p_holm_core9:.3f}')
        if scale==100:ax.set_ylabel('Percentage points/trial' if m=='slope' else 'Percent / percentage points')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'paired_learning_changes.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,tag in zip(axes,['pre','post','delta']):
        matrix=np.eye(9);stars=np.zeros((9,9),bool)
        for i,j in combinations(range(9),2):
            x=f'{METRICS[i]}_{tag}' if tag!='delta' else f'delta_{METRICS[i]}'
            y=f'{METRICS[j]}_{tag}' if tag!='delta' else f'delta_{METRICS[j]}'
            a=primary[(primary.x==x)&(primary.y==y)]
            if a.empty:a=primary[(primary.y==x)&(primary.x==y)]
            if not a.empty:
                r=a.iloc[0];matrix[i,j]=matrix[j,i]=r.rho;stars[i,j]=stars[j,i]=r.q_domain<.05
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        for i in range(9):
            for j in range(9):
                if i!=j:ax.text(j,i,f'{matrix[i,j]:.2f}'+('*' if stars[i,j] else ''),ha='center',va='center',fontsize=7)
        ax.set(xticks=range(9),yticks=range(9),xticklabels=METRICS,yticklabels=METRICS,title=tag)
        ax.tick_params(axis='x',rotation=90)
    fig.colorbar(im,ax=axes,label='Spearman ρ',shrink=.7)
    fig.suptitle(f'{len(d)}-participant cohort; measure-specific outlier removal. * BH q<.05 across 351 core tests')
    fig.savefig(out/'learning_correlation_heatmaps.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    rng=np.random.default_rng(8102)
    for ses,color,label in [(1,'#287c92','Before VR'),(2,'#b75b3c','After VR')]:
        a=curves[(curves['mode']=='rt200_primary')&(curves.session==ses)].pivot(index='subject_id',columns='trial',values='optimal')
        y=a.to_numpy();boot=np.nanmean(y[rng.integers(0,len(y),(3000,len(y)))],axis=1)
        low,high=np.nanquantile(boot,[.025,.975],axis=0)
        ax.plot(a.columns,100*np.nanmean(y,axis=0),'o-',color=color,label=label)
        ax.fill_between(a.columns,100*low,100*high,color=color,alpha=.15)
    ax.set(xlabel='Original trial position within block',ylabel='Optimal choices (%)',xticks=[1,6,12,18],
           title=f'Group learning curves (n={len(d)}); participant bootstrap 95% CIs')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'group_average_curves.png',dpi=220);plt.close(fig)
    specs=[('delta_winstay','delta_accuracy'),('delta_winstay','delta_slope'),
           ('delta_loseswitch','delta_accuracy'),('winstay_post','slope_post'),
           ('loseswitch_post','slope_post'),('speech_math_hr_reactivity','delta_k')]
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,(x,y) in zip(axes.flat,specs):
        rows=primary[((primary.x==x)&(primary.y==y))|((primary.y==x)&(primary.x==y))]
        if rows.empty:
            ax.axis('off');ax.text(.5,.5,'Supplemental HR not requested / unavailable',ha='center',va='center',wrap=True)
            continue
        r=rows.iloc[0]
        ids=[int(i) for i in r.participant_ids.split(',')];a=d.loc[ids,[x,y]]
        ax.scatter(a[x],a[y],color='#287c92',alpha=.8)
        ax.set(xlabel=x,ylabel=y,title=f'ρ={r.rho:.2f}, n={int(r.n)}, BH q={r.q_domain:.3f}')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exploratory associations; HR panel uses separate supplemental correction')
    fig.savefig(out/'strategy_learning_and_supplemental_hr_scatterplots.png',dpi=220);plt.close(fig)


def write_readable_summary(out):
    """Generate narrative text/Markdown reports directly from current tables."""
    from learning_analysis_reports import write_learning_reports
    write_learning_reports(out)



if __name__=='__main__':main()
