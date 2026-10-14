#!/usr/bin/env python3
"""Conservative, reproducible AF7/AF8 raw-EEG audit for Puberty_VR.

Run: python3 scripts/EEG_Audit.py
Dependencies: numpy pandas scipy matplotlib

This script flags possible artifacts; it cannot prove neural origin or diagnose stress.
No synthetic waveform reconstruction, ICA, or interpolation across missing EEG spans.
"""
from __future__ import annotations
import argparse
import io
import json
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

FS=256.0
CHANNELS=('AF7','AF8')
RAW=('RAW_AF7','RAW_AF8')
WINDOW_SEC=4.0
STEP_SEC=2.0
BANDS={'slow_1_4_warning':(1.,4.),'alpha':(8.,13.),'beta':(13.,30.),'high_beta':(20.,30.),'hf_warning':(35.,45.)}
VERSION='2.0'


def discover(root:Path):
    # Subject directory can itself be root; recursively supports nested CSV and ZIP.
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name.startswith('._') or '__MACOSX' in p.parts or p.name.startswith('.'):
            continue
        if not re.search(r'eeg', p.name, re.I) or re.search(r'(?:_Result|_summary|_qc|_audit)\.csv$',p.name,re.I):
            continue
        if p.suffix.lower() in ('.csv','.zip'):
            yield p


def load(path:Path):
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/') and not Path(n).name.startswith('._')]
            if len(names)!=1:
                raise ValueError(f'Expected one CSV within ZIP; found {len(names)}')
            with z.open(names[0]) as f:
                df=pd.read_csv(f,low_memory=False)
    else:
        df=pd.read_csv(path,low_memory=False)
    required=['TimeStamp',*RAW]
    missing=[c for c in required if c not in df.columns]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    # Only rows with at least one actual frontal raw sample are EEG rows.
    v=df[list(RAW)].apply(pd.to_numeric,errors='coerce')
    mask=v.notna().any(axis=1)
    x=v.loc[mask].reset_index(drop=True).to_numpy(dtype=float)
    times=pd.to_datetime(df.loc[mask,'TimeStamp'],errors='coerce')
    valid_time=times.notna().to_numpy()
    x=x[valid_time]
    t=times.loc[valid_time].reset_index(drop=True)
    if len(t)<2:raise ValueError('Insufficient timestamped raw EEG samples')
    seconds=(t-t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    # A few exports contain a leading metadata/late timestamp, remove out of order samples,
    # keeping original acquisition order rather than sorting to mask clock resets.
    good=np.r_[True,np.diff(seconds)>0]
    removed=int(np.count_nonzero(~good))
    seconds=seconds[good];x=x[good]
    seconds=seconds-seconds[0]
    return seconds,x,int(len(df)),removed


def identify_spans(t, gap_factor=3.5):
    # Treat any interruption exceeding 3.5 nominal sample periods as discontinuity.
    # No filtering or epochs are allowed to cross those boundaries.
    jump=np.flatnonzero(np.diff(t)>gap_factor/FS)+1
    return [(int(s),int(e)) for s,e in zip(np.r_[0,jump],np.r_[jump,len(t)])], int(len(jump))


def band_integral(f,p,lo,hi):
    # Integrate with interpolated boundaries to avoid overlapping 13 Hz bins.
    sel=(f>lo)&(f<hi)
    x=np.r_[lo,f[sel],hi]
    y=np.interp(x,f,p)
    return float(np.trapezoid(y,x))


def powers(x, hp=0.5):
    # epoch has no missing values. The window is 4s and periodogram uses Hann taper.
    # 1s padding outside epoch is intentionally NOT applied.
    f,p=signal.welch(x,fs=FS,window='hann',nperseg=len(x),noverlap=0,
                    detrend='constant',scaling='density')
    vals={n:band_integral(f,p,*b) for n,b in BANDS.items()}
    vals['ratio']=vals['beta']/vals['alpha'] if vals['alpha']>0 else np.nan
    vals['hf_beta_indicator']=vals['hf_warning']/vals['beta'] if vals['beta']>0 else np.nan
    return vals,f,p


def preprocess_epoch(raw, hp=0.5, lp=45.0):
    # Zero-phase Butterworth; notch is unnecessary when low-pass is 45 Hz and
    # the 60Hz component is excluded. Optional 60Hz notch would be redundant.
    sos=signal.butter(4,[hp,lp],btype='bandpass',fs=FS,output='sos')
    return signal.sosfiltfilt(sos,raw)


def quality(raw, filtered, params, baseline):
    reasons=[];warnings_list=[]
    if not np.isfinite(raw).all():reasons.append('missing_samples')
    if len(raw)<int(FS*WINDOW_SEC):reasons.append('short_window')
    finite=raw[np.isfinite(raw)]
    if not len(finite):return ['missing_samples'],[]
    if np.nanstd(finite)<params.flat_std:reasons.append('flatline')
    # Hardware boundaries are unverified. Repeated 0 and 1450 are heuristic sentinels.
    edge=np.mean(np.isclose(finite,0,atol=0.01)|np.isclose(finite,1450,atol=0.01))
    if edge>params.bound_fraction:reasons.append('boundary_values')
    if np.any(np.abs(finite)>params.absolute_limit):
        warnings_list.append('large_raw_amplitude_unverified_units')
    # Long repeated near-identical runs signal dropout even if window variance normal.
    d=np.abs(np.diff(finite));same=d<0.01
    if len(same):
        run=np.diff(np.r_[0,np.flatnonzero(~same)+1,len(same)]).max(initial=0)
        if run>=int(params.flat_run_ms/1000*FS):reasons.append('constant_run')
    if np.isfinite(filtered).all():
        amp=np.ptp(filtered)
        if amp>baseline['ptp_limit']:
            warnings_list.append('large_filtered_peak_to_peak')
        jump=np.abs(np.diff(raw))
        if np.any(jump>baseline['jump_limit']):
            warnings_list.append('abrupt_step_possible_contact_artifact')
    return reasons,warnings_list


def audit_one(path:Path,outdir:Path, args):
    subject=(next((part for part in path.parts if re.fullmatch(r'sub\d+',part,re.I)),path.parent.name))
    phase=re.sub(r'^\d+_','',path.stem,flags=re.I).replace('_eeg','')
    base={'subject':subject,'phase':phase,'recording':path.name,'source':str(path),'audit_version':VERSION}
    try:
        t,x,nrows,nonmonotonic=load(path)
        dt=np.diff(t);duration=float(t[-1]-t[0]); observed=(len(t)-1)/duration if duration>0 else 0
        med_rate=1/np.median(dt[dt>0]) if np.any(dt>0) else np.nan
        # Median captures high rate; average protects against huge missing spans.
        is_full= 230 <= med_rate <= 290 and observed>=args.min_average_rate and len(t)>=4*FS
        info={**base,'status':'SKIPPED_LOW_RATE' if not is_full else 'AUDITED',
              'raw_export_rows':nrows,'raw_sample_rows':len(t),'duration_s':duration,
              'estimated_median_hz':med_rate,'observed_average_hz':observed,
              'out_of_order_or_duplicate_rows':nonmonotonic}
        if not is_full:return info,[],[]
        spans,ngaps=identify_spans(t)
        info['timestamp_gaps']=ngaps
        steps=int(STEP_SEC*FS); win=int(WINDOW_SEC*FS)
        rows=[];raw_cache=[]
        for s,e in spans:
            for start in range(s,e-win+1,steps):
                end=start+win
                # Verify span timing: sampling jitter is tolerated, but no dropouts.
                local_dt=np.diff(t[start:end]); span_duration=t[end-1]-t[start]
                timing_ok=(np.count_nonzero(local_dt>args.gap_factor/FS)==0 and
                           abs(span_duration-(win-1)/FS)<args.window_duration_tolerance)
                # Keep both channels in common windows; never interpolate sample gaps.
                for ci,ch in enumerate(CHANNELS):
                    r=x[start:end,ci]
                    row={**base,'channel':ch,'start_s':float(t[start]),'end_s':float(t[end-1]),
                         'window_index':start, 'timing_ok':bool(timing_ok)}
                    why=[]; caution=[]
                    if not timing_ok:why.append('irregular_timestamps')
                    if np.any(~np.isfinite(r)):why.append('missing_samples')
                    clean=None
                    if not why:
                        try:
                            clean=preprocess_epoch(r)
                            why,caution=quality(r,clean,args,{ 'ptp_limit':args.ptp_warning,
                                                               'jump_limit':args.jump_warning})
                            metrics,_,_=powers(clean)
                            row.update(metrics)
                            # Sensitivity: slightly different accepted passband.
                            alternate=preprocess_epoch(r,hp=1.0,lp=40.0)
                            alt_metrics,_,_=powers(alternate)
                            row['alternate_beta_alpha_ratio']=alt_metrics['ratio']
                            if np.isfinite(metrics['ratio']) and metrics['ratio']>0:
                                row['relative_filter_sensitivity']=abs(alt_metrics['ratio']/metrics['ratio']-1)
                                if row['relative_filter_sensitivity']>args.filter_change_warning:
                                    caution.append('filter_sensitive_ratio')
                        except (ValueError,FloatingPointError) as exc:why.append('filter_error:'+str(exc))
                    if len(r):
                        finite=r[np.isfinite(r)]
                        row['raw_ptp']=float(np.ptp(finite)) if len(finite) else np.nan
                        row['raw_max_jump']=float(np.max(np.abs(np.diff(finite)))) if len(finite)>1 else np.nan
                        row['boundary_fraction_observed']=float(np.mean(np.isclose(finite,0,atol=.01)|np.isclose(finite,1450,atol=.01))) if len(finite) else np.nan
                        row['filtered_ptp']=float(np.ptp(clean)) if clean is not None else np.nan
                    row.update({'hard_reasons':'|'.join(sorted(set(why))),
                                'warning_reasons':'|'.join(sorted(set(caution))),
                                'hard_pass':len(why)==0})
                    rows.append(row)

        if not rows:
            info.update(status='NO_CONTIGUOUS_WINDOWS',windows_total=0)
            return info,[],[]
        df=pd.DataFrame(rows)
        # Post hoc HF and beta outlier warning by channel (do not auto remove: possible neural events).
        for ch in CHANNELS:
            m=(df.channel==ch)&df.hard_pass
            for feature in ['hf_beta_indicator','slow_1_4_warning','beta','alpha']:
                vals=df.loc[m,feature].astype(float)
                if len(vals)<8:continue
                positive=vals[vals>0]
                if len(positive)<8:continue
                lv=np.log(positive);median=np.median(lv);mad=1.4826*np.median(np.abs(lv-median))
                if mad<=1e-8:continue
                bad=m & (np.log(df[feature].clip(lower=1e-20))>median+args.outlier_mad*mad)
                if bad.any():
                    df.loc[bad,'warning_reasons']=df.loc[bad,'warning_reasons'].map(lambda a:(a+'|' if a else '')+('possible_muscle_artifact' if feature=='hf_beta_indicator' else 'possible_ocular_or_slow_artifact' if feature=='slow_1_4_warning' else 'unusual_'+feature))
        # Candidate strict sensitivity screening removes any window with a warning.
        df['strict_pass']=df.hard_pass & df.warning_reasons.eq('')
        # pooled frontal results only if BOTH channels in same windows pass.
        common=df.pivot(index='window_index',columns='channel',values='hard_pass').fillna(False).all(axis=1)
        common_ids=set(common.index[common].tolist())
        df['paired_pass']=df.hard_pass & df.window_index.isin(common_ids)
        info.update(windows_total=int(len(df)//2),af7_pass=int(((df.channel=='AF7')&df.hard_pass).sum()),
                    af8_pass=int(((df.channel=='AF8')&df.hard_pass).sum()),
                    paired_pass_windows=int(len(common_ids)),
                    af7_strict_pass=int(((df.channel=='AF7')&df.strict_pass).sum()),
                    af8_strict_pass=int(((df.channel=='AF8')&df.strict_pass).sum()))
        outdir.mkdir(parents=True,exist_ok=True)
        fig,axs=plt.subplots(4,1,figsize=(15,11),layout='constrained')
        for ci,ch in enumerate(CHANNELS):
            # Plot sparse original raw trace for context; units not assumed.
            axs[ci].plot(t[::max(1,len(t)//12000)],x[::max(1,len(t)//12000),ci],lw=.35,color=['#32669a','#a15e3d'][ci])
            axs[ci].set(title=f'{ch} raw values (uncalibrated units)',xlabel='Elapsed seconds',ylabel='RAW')
            part=df[df.channel==ch]
            for _,rr in part.iterrows():
                if not rr.hard_pass:axs[ci].axvspan(rr.start_s,rr.end_s,color='red',alpha=.09)
            for band,sty in [('alpha','-'),('beta','--')]:
                axs[2+ci].plot(part.start_s+WINDOW_SEC/2,part[band].where(part.hard_pass),sty,lw=.9,label=band+' (hard-pass only)')
            axs[2+ci].set(title=f'{ch} filtered 4-s epoch band powers (hard-pass only)',xlabel='Elapsed seconds',ylabel='Power (raw units squared)',yscale='log')
            axs[2+ci].legend()
        fig.suptitle(f'{subject} | {phase} | red shaded = hard rejected windows',fontsize=14)
        fig.savefig(outdir/(path.stem+'_audit.png'),dpi=130)
        plt.close(fig)
        reports=[]
        for ch in CHANNELS:
            for label,maskcol in [('primary','hard_pass'),('strict','strict_pass'),('paired','paired_pass')]:
                part=df[(df.channel==ch)&df[maskcol]]
                alpha=float(part.alpha.mean()) if len(part) else np.nan
                beta=float(part.beta.mean()) if len(part) else np.nan
                reports.append({**base,'channel':ch,'method':label,'accepted_windows':len(part),
                                'accepted_window_seconds_overlapping':len(part)*WINDOW_SEC,
                                'alpha_power_mean':alpha,'beta_power_mean':beta,
                                'high_beta_power_mean':float(part.high_beta.mean()) if len(part) else np.nan,
                                'beta_alpha_ratio_of_mean_powers':beta/alpha if alpha>0 else np.nan,
                                'log_beta_alpha_ratio':np.log(beta/alpha) if beta>0 and alpha>0 else np.nan,
                                'hf_indicator_median':float(part.hf_beta_indicator.median()) if len(part) else np.nan})
        return info,df.to_dict('records'),reports
    except Exception as exc:
        return {**base,'status':'ERROR','error':f'{type(exc).__name__}: {exc}'},[],[]




def accepted_coverage_seconds(part):
    """Union length of overlapping accepted windows: no double-counting."""
    intervals=sorted((float(a),float(b)) for a,b in zip(part.start_s,part.end_s))
    if not intervals:return 0.0
    a,b=intervals[0];total=0.
    for c,d in intervals[1:]:
        if c<=b:b=max(b,d)
        else:total+=b-a;a,b=c,d
    return float(total+b-a)


def write_review_gallery(source, record, output):
    """Show representative raw AF7/AF8 epochs across flagged and passing groups."""
    try:t,x,_,_=load(Path(source))
    except Exception:return
    groups=[('hard_reject',record[~record.hard_pass]),
            ('warning_only',record[record.hard_pass & ~record.strict_pass]),
            ('accepted',record[record.strict_pass])]
    choices=[]
    for label,subset in groups:
        pairs=subset[['window_index','start_s']].drop_duplicates().head(2)
        for _,rr in pairs.iterrows():choices.append((label,int(rr.window_index),float(rr.start_s)))
    if not choices:return
    fig,axs=plt.subplots(len(choices),1,figsize=(12,max(4,2.1*len(choices))),squeeze=False,layout='constrained')
    for ax,(label,start,when) in zip(axs[:,0],choices):
        xx=x[start:start+int(FS*WINDOW_SEC)]
        ts=t[start:start+len(xx)]
        if len(xx)>0:
            for j,ch in enumerate(CHANNELS):ax.plot(ts,xx[:,j],lw=.65,label=ch)
        ax.set(title=f'{label}: {when:.2f}s (unfiltered raw signal)',ylabel='Unverified raw units',xlabel='Elapsed seconds')
        ax.legend(fontsize=8,loc='upper right')
    fig.savefig(output,dpi=110);plt.close(fig)


def create_second_pass(out,args,root):
    """Post-audit QC: empirical threshold diagnostics, review, and candidate powers."""
    wd=out/'eeg_window_qc_and_power.csv'; rd=out/'eeg_recording_qc.csv'
    if not wd.exists() or wd.stat().st_size==0:return
    w=pd.read_csv(wd,low_memory=False,keep_default_na=False)
    if w.empty:return
    for c in ['start_s','end_s','alpha','beta','high_beta','hf_warning','ratio','hf_beta_indicator',
              'filtered_ptp','raw_ptp','raw_max_jump','boundary_fraction_observed','relative_filter_sensitivity']:
        if c in w:w[c]=pd.to_numeric(w[c],errors='coerce')
    for c in ['hard_pass','strict_pass','paired_pass']:
        w[c]=w[c].astype(str).str.lower().eq('true')
    w['recording_id']=w.subject.astype(str)+'/'+w.recording.astype(str)
    # Robust empirical thresholds are diagnostics only; do not use to silently redefine QC.
    diagnostics=[]
    for ch,group in w.groupby('channel'):
        for metric in ['raw_ptp','filtered_ptp','raw_max_jump','boundary_fraction_observed','hf_beta_indicator','relative_filter_sensitivity']:
            if metric not in group:continue
            v=group.loc[group.hard_pass,metric].replace([np.inf,-np.inf],np.nan).dropna()
            if len(v)==0:continue
            for pct in [50,75,90,95,99]:
                diagnostics.append({'channel':ch,'metric':metric,'percentile':pct,'value':float(np.percentile(v,pct)),
                                    'n_hard_pass_windows':len(v),'note':'Descriptive, NOT a validated artifact cutoff'})
    pd.DataFrame(diagnostics).to_csv(out/'eeg_threshold_diagnostics.csv',index=False)
    # Illustrate dependence on provisional cutoffs WITHOUT overriding core exclusions.
    sweeps=[]
    for bound in [.005,.01,.02,.05]:
        for ptp in [300.,500.,800.,1200.]:
            subset=w.copy()
            if 'boundary_fraction_observed' not in subset:continue
            # This is a deliberately partial what-if, not a corrected re-audit.
            boundflag=subset.boundary_fraction_observed>bound
            ptpflag=subset.filtered_ptp>ptp
            for ch in CHANNELS:
                z=subset[subset.channel==ch]
                sweeps.append({'channel':ch,'hypothetical_bound_fraction':bound,
                  'hypothetical_ptp_warning':ptp,
                  'windows_with_boundary_flag':int(boundflag.loc[z.index].sum()),
                  'windows_with_large_ptp_warning':int(ptpflag.loc[z.index].sum()),
                  'note':'Partial sensitivity summary only; thresholds need human/device validation'})
    pd.DataFrame(sweeps).to_csv(out/'eeg_threshold_sensitivity.csv',index=False)
    # Recording retention, reason breakdown and conservative recommendations.
    rec=[]
    for (sid,name),q in w.groupby(['subject','recording'],sort=True):
        total=int(q.window_index.nunique())
        counts={ch:int(q[(q.channel==ch)&q.hard_pass].shape[0]) for ch in CHANNELS}
        paired=q[q.paired_pass];paired_count=int(paired.window_index.nunique())
        pa=paired_count/total if total else 0
        flags=q.hard_reasons.fillna('').astype(str).str.split('|').explode()
        warnings=q.warning_reasons.fillna('').astype(str).str.split('|').explode()
        note='HIGH_RETENTION_REVIEW' if pa>=.70 else 'LOW_RETENTION_INVESTIGATE' if pa<.40 else 'MODERATE_RETENTION_REVIEW'
        rec.append({'subject':sid,'phase':q.phase.iloc[0],'recording':name,'source':q.source.iloc[0],
                    'windows_per_channel':total,'AF7_hard_pass':counts['AF7'],'AF8_hard_pass':counts['AF8'],
                    'paired_hard_pass':paired_count,'paired_retention_fraction':pa,
                    'paired_unique_coverage_s':accepted_coverage_seconds(paired.drop_duplicates('window_index')),
                    'priority':note,
                    'top_hard_flags':'|'.join(f'{k}:{v}' for k,v in flags.value_counts().items() if k)[:350],
                    'top_warnings':'|'.join(f'{k}:{v}' for k,v in warnings.value_counts().items() if k)[:350]})
    inventory=pd.DataFrame(rec)
    inventory.to_csv(out/'eeg_review_priorities.csv',index=False)
    review_path=out/'eeg_manual_review.csv'
    if args.manual_review:
        decisions=pd.read_csv(args.manual_review,dtype=str).fillna('')
    elif review_path.exists():
        decisions=pd.read_csv(review_path,dtype=str).fillna('')
    else:decisions=pd.DataFrame()
    # Never auto-approve; retain user comments across reruns.
    cols=['subject','recording','channel','review_status','reviewer','review_date','notes']
    existing={}
    if all(c in decisions.columns for c in cols):
        for _,r in decisions.iterrows():existing[(r.subject,r.recording,r.channel)]=r.to_dict()
    fresh=[]
    for _,r in inventory.iterrows():
        for ch in ['AF7','AF8','BOTH']:
            key=(str(r.subject),str(r.recording),ch)
            fresh.append(existing.get(key,{'subject':key[0],'recording':key[1],
                'channel':ch,'review_status':'PENDING','reviewer':'','review_date':'','notes':''}))
    pd.DataFrame(fresh,columns=cols).to_csv(review_path,index=False)
    approvals={(str(r['subject']),str(r['recording']),r['channel']):r['review_status'].strip().upper()
               for r in fresh}
    # Analysis outputs: powers based on ACCEPTED windows; quality status remains separate.
    final=[]
    for (sid,recname),q in w.groupby(['subject','recording'],sort=True):
        for channel in ['AF7','AF8','BOTH']:
            for strategy in ['primary','strict']:
                if channel=='BOTH':
                    mask=q.paired_pass if strategy=='primary' else q.strict_pass & q.paired_pass
                    if strategy=='strict':
                        ok=q[mask].groupby('window_index').channel.nunique()
                        qsel=q[mask & q.window_index.isin(ok[ok==2].index)]
                    else:qsel=q[mask]
                    # One observation per timestamp: average AF7/AF8 powers before ratio.
                    ps=qsel.groupby('window_index',as_index=False)[['alpha','beta','high_beta','start_s','end_s']].mean()
                else:
                    qsel=q[(q.channel==channel)&q[strategy.replace('primary','hard_pass').replace('strict','strict_pass')]]
                    ps=qsel[['alpha','beta','high_beta','start_s','end_s']].copy()
                ps=ps.replace([np.inf,-np.inf],np.nan).dropna(subset=['alpha','beta'])
                ps=ps[(ps.alpha>0)&(ps.beta>=0)]
                a=float(ps.alpha.mean()) if len(ps) else np.nan
                b=float(ps.beta.mean()) if len(ps) else np.nan
                ratio=b/a if np.isfinite(a) and a>0 else np.nan
                status=approvals.get((str(sid),str(recname),channel),'PENDING')
                if channel=='BOTH' and not all(approvals.get((str(sid),str(recname),c))=='APPROVED' for c in ['AF7','AF8','BOTH']):
                    status='PENDING_CHANNEL_REVIEW'
                final.append({'subject':sid,'phase':q.phase.iloc[0],'recording':recname,'channel':channel,
                              'strategy':strategy,'n_windows':len(ps),
                              'unique_coverage_s':accepted_coverage_seconds(ps),
                              'alpha_power':a,'beta_power':b,
                              'high_beta_power':float(ps.high_beta.mean()) if len(ps) else np.nan,
                              'beta_alpha_ratio':ratio,'log_beta_alpha_ratio':float(np.log(ratio)) if ratio>0 else np.nan,
                              'manual_review_status':status,
                              'interpretation_status':'REVIEWED_CANDIDATE' if status=='APPROVED' and len(ps)>0 else 'NOT_VALIDATED'})
    f=pd.DataFrame(final)
    f.to_csv(out/'eeg_final_candidate_band_powers.csv',index=False)
    # Paired pre/post changes are subject-level; no p-values on overlapping windows.
    primary=f[(f.strategy=='primary')&(f.n_windows>0)].copy()
    before=primary[primary.phase=='pre_vr'];after=primary[primary.phase=='post_vr']
    pair=before.merge(after,on=['subject','channel'],suffixes=('_pre','_post'))
    change=[]
    for _,r in pair.iterrows():
        vals={'subject':r.subject,'channel':r.channel,'pre_recording':r.recording_pre,
              'post_recording':r.recording_post,
              'both_manual_approved':r.manual_review_status_pre=='APPROVED' and r.manual_review_status_post=='APPROVED'}
        for band in ['alpha_power','beta_power','beta_alpha_ratio']:
            a=r[band+'_pre'];b=r[band+'_post']
            vals[band+'_pre']=a;vals[band+'_post']=b
            vals[band+'_percent_change']=100*(b/a-1) if a>0 else np.nan
        change.append(vals)
    pd.DataFrame(change,columns=['subject','channel','pre_recording','post_recording','both_manual_approved',
      *[n+s for n in ['alpha_power','beta_power','beta_alpha_ratio'] for s in ['_pre','_post','_percent_change']]]).to_csv(out/'eeg_pre_post_candidate_comparisons.csv',index=False)
    if not args.skip_review_figures:
        figroot=out/'review_figures'
        for _,r in inventory.iterrows():
            folder=figroot/str(r.subject);folder.mkdir(parents=True,exist_ok=True)
            rec_w=w[(w.subject==r.subject)&(w.recording==r.recording)]
            write_review_gallery(r.source,rec_w,folder/(Path(r.recording).stem+'_review.png'))
    print('Second-pass reports: thresholds, priorities, manual-review template, candidate power and pre/post tables.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subjects-dir',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--subject',default=None,help='Optional subject number, e.g. 029')
    p.add_argument('--min-average-rate',type=float,default=180.,help='Reject recordings with very sparse EEG data')
    p.add_argument('--gap-factor',type=float,default=3.5)
    p.add_argument('--window-duration-tolerance',type=float,default=.08)
    p.add_argument('--flat-std',type=float,default=.05)
    p.add_argument('--flat-run-ms',type=float,default=125.)
    p.add_argument('--bound-fraction',type=float,default=.01,help='Suspect 0/1450 occupancy per epoch (heuristic)')
    p.add_argument('--absolute-limit',type=float,default=2500.,help='Warning only; units not verified')
    p.add_argument('--ptp-warning',type=float,default=500.,help='Warning only; units not verified')
    p.add_argument('--jump-warning',type=float,default=300.,help='Warning only; units not verified')
    p.add_argument('--outlier-mad',type=float,default=5.)
    p.add_argument('--filter-change-warning',type=float,default=.25)
    p.add_argument('--manual-review',type=Path,default=None,help='Optional reviewer decisions CSV; do not edit original raw data')
    p.add_argument('--skip-review-figures',action='store_true',help='Skip detailed per-recording review galleries')
    a=p.parse_args()
    # Works whether installed in scripts/ or scripts/eeg/.
    parents=Path(__file__).resolve().parents
    project=next((q for q in parents if (q/'data'/'raw'/'subject_data').is_dir()),parents[2] if len(parents)>2 else parents[1])
    root=a.subjects_dir or project/'data'/'raw'/'subject_data'
    if not root.is_dir():
        p.error(f'Subject folder not found: {root}. Pass --subjects-dir.')
    out=a.output_dir or project/'outputs'/'eeg'/'audit'
    paths=list(discover(root))
    if a.subject:paths=[f for f in paths if f'sub{a.subject.zfill(3)}' in f.parts]
    if not paths:p.error('No EEG CSV/ZIP inputs were discovered.')
    infos=[];windows=[];summary=[]
    out.mkdir(parents=True,exist_ok=True)
    for i,path in enumerate(paths,1):
        sub=next((v for v in path.parts if re.fullmatch(r'sub\d+',v)),path.parent.name)
        print(f'[{i}/{len(paths)}] {sub}/{path.name}',flush=True)
        info,window_rows,summaries=audit_one(path,out/'figures'/sub,a)
        infos.append(info);windows.extend(window_rows);summary.extend(summaries)
        print('  ->',info['status'],f"({info.get('paired_pass_windows','-')} paired windows)",flush=True)
    pd.DataFrame(infos).to_csv(out/'eeg_recording_qc.csv',index=False)
    pd.DataFrame(windows).to_csv(out/'eeg_window_qc_and_power.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'eeg_frontal_band_summary.csv',index=False)
    # Explicit pairing only; no inferential tests on windows as independent observations.
    sums=pd.DataFrame(summary)
    if not sums.empty:
        prim=sums[(sums.method=='paired') & sums.channel.eq('AF7')].copy()
        prim=prim[['subject','phase','recording','accepted_windows']]
        prim.to_csv(out/'eeg_paired_windows_inventory.csv',index=False)
    settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    (out/'audit_settings.json').write_text(json.dumps({'version':VERSION,'fs_assumed':FS,'window_s':WINDOW_SEC,'step_s':STEP_SEC,'bands_hz':BANDS,'settings':settings},indent=2))
    create_second_pass(out, a, root)
    print('\nReports saved in:',out)
    print(pd.DataFrame(infos).status.value_counts().to_string())

if __name__=='__main__':main()
