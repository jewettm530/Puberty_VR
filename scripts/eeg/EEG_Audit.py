#!/usr/bin/env python3
"""Conservative, reproducible AF7/AF8 raw-EEG audit for Puberty_VR.

Run: python3 scripts/eeg/EEG_Audit.py
Dependencies: numpy pandas scipy matplotlib

This script flags possible artifacts; it cannot prove neural origin or diagnose stress.
No synthetic waveform reconstruction, ICA, or interpolation across missing EEG spans.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS=256.0
CHANNELS=('AF7','AF8')
RAW=('RAW_AF7','RAW_AF8')
WINDOW_SEC=2.0
STEP_SEC=1.0
BANDS={'slow_1_4_warning':(1.,4.),'alpha':(8.,13.),'beta':(13.,30.),'high_beta':(20.,30.),'hf_warning':(35.,45.)}
VERSION='4.2-eyes-open-welch2s-original-layout'
PREPROCESSOR = None  # loaded once by main(); owns filtering and PSD calculations


def discover(root:Path):
    # Subject directory can itself be root; recursively supports nested CSV and ZIP.
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name.startswith('._') or '__MACOSX' in p.parts or p.name.startswith('.'):
            continue
        if not re.search(r'eeg', p.name, re.I) or re.search(r'(?:_Result|_summary|_qc|_audit)\.csv$',p.name,re.I):
            continue
        if p.suffix.lower() in ('.csv','.zip'):
            yield p


def load(path: Path):
    """Compatibility loader using the shared timestamp and sample parser."""
    processor = PREPROCESSOR or load_preprocessing_module(Path(__file__).resolve().parents[2])
    data = processor.load_recording(path)
    return data['times'], data['raw'], data['raw_export_rows'], data['out_of_order_or_duplicate_rows']

def identify_spans(t, gap_factor=3.5):
    # Treat any interruption exceeding 3.5 nominal sample periods as discontinuity.
    # No filtering or epochs are allowed to cross those boundaries.
    jump=np.flatnonzero(np.diff(t)>gap_factor/FS)+1
    return [(int(s),int(e)) for s,e in zip(np.r_[0,jump],np.r_[jump,len(t)])], int(len(jump))


def powers(filtered):
    """Fixed two-second Welch PSD and integrated bands."""
    frequencies, psd = PREPROCESSOR.compute_psd(np.asarray(filtered), noverlap=0)
    bands = {'theta': PREPROCESSOR.BANDS['theta'], 'alpha': (8.,13.),
             'beta': (13.,30.), 'high_beta': (20.,30.),
             'hf_warning': (35.,45.), 'slow_1_4_warning': (1.,4.)}
    values = {name: float(PREPROCESSOR.band_power(frequencies, psd[:,None], lo, hi)[0])
              for name,(lo,hi) in bands.items()}
    values['ratio'] = values['beta']/values['alpha'] if values['alpha'] > 0 else np.nan
    values['hf_beta_indicator'] = values['hf_warning']/values['beta'] if values['beta'] > 0 else np.nan
    return values, frequencies, psd

def prefilter_valid_spans(raw, spans):
    return PREPROCESSOR.prefilter_valid_spans(raw, spans)

def audit_one(path: Path, outdir: Path, args, pre_result=None):
    """Audit the exact saved preprocessed samples and epoch decisions."""
    processor = PREPROCESSOR
    subject, phase = processor.recording_identity(path)
    base = {'subject':'sub'+subject, 'phase':phase, 'recording':path.name,
            'source':str(path), 'audit_version':VERSION}
    if pre_result is None:
        raise ValueError("Run preprocessing first and pass its saved output to audit_one")
    status = pre_result['preprocessing_status']
    if status != 'COMPLETED':
        return {**base, 'status':status, 'error':pre_result['preprocessing_error']}, [], []
    result = processor.load_preprocessed(Path(pre_result['preprocessed_path']))
    meta, t, x = result['metadata'], result['times'], result['filtered_uv']
    df = result['windows'].copy()
    for col in ('hard_pass','automatic_pass','paired_pass','strict_pass','timing_ok'):
        df[col] = df[col].astype(str).str.lower().eq('true')
    for col in processor.QC_COLUMNS:
        if col not in ('channel','hard_reasons','warning_reasons','hard_pass',
                       'automatic_pass','paired_pass','strict_pass','timing_ok'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for key,value in base.items():
        df[key] = value
    df['preprocessed_path'] = pre_result['preprocessed_path']
    df['preprocessing_id'] = meta['preprocessing_id']
    df['selection'] = meta['selection']
    df['analysis_id'] = meta['analysis_id']
    df['alternate_beta_alpha_ratio'] = np.nan
    df['relative_filter_sensitivity'] = np.nan
    df['warning_reasons'] = df['warning_reasons'].fillna('').astype(str)
    # These additional diagnostics do not change primary preprocessing decisions.
    for channel in CHANNELS:
        mask = (df.channel == channel) & df.hard_pass
        for feature in ('hf_beta_indicator','slow_1_4_warning','beta','alpha'):
            values = df.loc[mask,feature].astype(float)
            positive = values[values > 0]
            if len(positive) < 8:
                continue
            logged = np.log(positive)
            center = np.median(logged)
            spread = 1.4826*np.median(np.abs(logged-center))
            if spread <= 1e-8:
                continue
            flagged = mask & (np.log(df[feature].clip(lower=1e-20)) > center+args.outlier_mad*spread)
            label = ('possible_muscle_artifact' if feature == 'hf_beta_indicator'
                     else 'possible_ocular_or_slow_artifact' if feature == 'slow_1_4_warning'
                     else 'unusual_'+feature)
            df.loc[flagged,'warning_reasons'] = df.loc[flagged,'warning_reasons'].map(
                lambda value: (value+'|' if value else '')+label)
    for flag, label in ((df.raw_ptp > args.ptp_warning, 'large_peak_to_peak'),
                        (df.raw_max_jump > args.jump_warning, 'abrupt_step_possible_contact_artifact')):
        df.loc[flag,'warning_reasons'] = df.loc[flag,'warning_reasons'].map(
            lambda value: (value+'|' if value else '')+label)
    df['strict_pass'] = df.hard_pass & df.warning_reasons.eq('')
    info = {**base, **{key:meta[key] for key in (
        'raw_export_rows','raw_sample_rows','selected_sample_rows',
        'estimated_median_hz','observed_average_hz','out_of_order_or_duplicate_rows',
        'timestamp_gaps','selection','analysis_start_s','analysis_end_s',
        'preprocessing_id','analysis_id','raw_offset_uv','amplitude_limit_uv')},
        'status':'AUDITED', 'duration_s':meta['analysis_end_s']-meta['analysis_start_s'],
        'windows_total':int(df.window_index.nunique()),
        'af7_pass':int(((df.channel=='AF7') & df.hard_pass).sum()),
        'af8_pass':int(((df.channel=='AF8') & df.hard_pass).sum()),
        'paired_pass_windows':int(df.loc[df.paired_pass,'window_index'].nunique()),
        'af7_strict_pass':int(((df.channel=='AF7') & df.strict_pass).sum()),
        'af8_strict_pass':int(((df.channel=='AF8') & df.strict_pass).sum()),
        'manual_bad_intervals':meta['manual_bad_intervals'],
        'ignored_annotation_files':'|'.join(meta['ignored_annotation_files'])}
    outdir.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(4,1,figsize=(15,11),layout='constrained')
    for ci,channel in enumerate(CHANNELS):
        axes[ci].plot(t,x[:,ci],lw=.3)
        axes[ci].axhline(meta['amplitude_limit_uv'],ls='--',color='red')
        axes[ci].axhline(-meta['amplitude_limit_uv'],ls='--',color='red')
        part = df[df.channel == channel]
        for row in part.loc[~part.hard_pass].itertuples():
            axes[ci].axvspan(row.start_s,row.end_s,color='red',alpha=.08)
        axes[ci].set(title=channel+' preprocessed',ylabel='µV',xlabel='Original elapsed seconds')
        for band,style in (('alpha','-'),('beta','--')):
            axes[2+ci].plot(part.start_s+WINDOW_SEC/2,part[band].where(part.hard_pass),
                           style,lw=.8,label=band)
        axes[2+ci].set(title=channel+' accepted 2-s Welch bands',yscale='log',ylabel='µV²')
        axes[2+ci].legend()
    fig.suptitle(f"{subject} | {phase} | {meta['selection']} | red = rejected")
    fig.savefig(outdir/(path.stem+'_audit.png'),dpi=130)
    plt.close(fig)
    summaries = []
    for channel in CHANNELS:
        for method,mask in (('primary','hard_pass'),('strict','strict_pass'),('paired','paired_pass')):
            part = df[(df.channel == channel) & df[mask]]
            alpha = float(part.alpha.mean()) if len(part) else np.nan
            beta = float(part.beta.mean()) if len(part) else np.nan
            summaries.append({**base,'channel':channel,'method':method,
                'accepted_windows':len(part),'accepted_window_seconds_overlapping':len(part)*WINDOW_SEC,
                'alpha_power_mean':alpha,'beta_power_mean':beta,
                'high_beta_power_mean':float(part.high_beta.mean()) if len(part) else np.nan,
                'beta_alpha_ratio_of_mean_powers':beta/alpha if alpha>0 else np.nan,
                'log_beta_alpha_ratio':np.log(beta/alpha) if beta>0 and alpha>0 else np.nan,
                'hf_indicator_median':float(part.hf_beta_indicator.median()) if len(part) else np.nan})
    return info, df.to_dict('records'), summaries

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
    """Show saved preprocessed samples on their original recording timeline."""
    if record.empty:
        return
    data = PREPROCESSOR.load_preprocessed(Path(record.preprocessed_path.iloc[0]))
    t, x = data['times'], data['filtered_uv']
    choices = []
    for label,subset in [('rejected',record[~record.hard_pass]),
                         ('warning',record[record.hard_pass & ~record.strict_pass]),
                         ('accepted',record[record.strict_pass])]:
        for row in subset[['window_index','start_s']].drop_duplicates().head(2).itertuples():
            choices.append((label,int(row.window_index)))
    fig,axes = plt.subplots(len(choices),1,figsize=(12,max(4,2*len(choices))),
                            squeeze=False,layout='constrained')
    for ax,(label,start) in zip(axes[:,0],choices):
        stop = start+PREPROCESSOR.WINDOW_LENGTH
        ax.plot(t[start:stop],x[start:stop])
        ax.set(title=label+' preprocessed window',xlabel='Original elapsed seconds',ylabel='µV')
    fig.savefig(output,dpi=110)
    plt.close(fig)

def create_second_pass(out,args,root):
    """Post-audit QC: empirical threshold diagnostics, review, and candidate powers."""
    wd=out/'eeg_window_qc_and_power.csv'; rd=out/'eeg_recording_qc.csv'
    if not wd.exists() or wd.stat().st_size==0:return
    w=pd.read_csv(wd,low_memory=False,keep_default_na=False)
    if w.empty:
        schemas = {
            'eeg_threshold_diagnostics.csv':['channel','metric','percentile','value'],
            'eeg_threshold_sensitivity.csv':['channel','hypothetical_bound_fraction','hypothetical_ptp_warning'],
            'eeg_review_priorities.csv':['subject','phase','recording','priority'],
            'eeg_final_candidate_band_powers.csv':['subject','phase','recording','channel','strategy','n_windows'],
            'eeg_pre_post_candidate_comparisons.csv':['subject','channel','pre_recording','post_recording'],
        }
        for filename,columns in schemas.items():
            pd.DataFrame(columns=columns).to_csv(out/filename,index=False)
        return
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
                    'analysis_id':q.analysis_id.iloc[0],'windows_per_channel':total,'AF7_hard_pass':counts['AF7'],'AF8_hard_pass':counts['AF8'],
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
    cols=['subject','recording','channel','review_status','reviewer','review_date','notes','analysis_id']
    existing={}
    if all(c in decisions.columns for c in cols[:-1]):
        for _,r in decisions.iterrows():existing[(r.subject,r.recording,r.channel)]=r.to_dict()
    fresh=[]
    for _,r in inventory.iterrows():
        for ch in ['AF7','AF8','BOTH']:
            key=(str(r.subject),str(r.recording),ch)
            decision = existing.get(key,{'subject':key[0],'recording':key[1],
                'channel':ch,'review_status':'PENDING','reviewer':'','review_date':'','notes':''}).copy()
            if decision.get('analysis_id') != r.analysis_id:
                decision['review_status'] = 'PENDING'
            decision['analysis_id'] = r.analysis_id
            fresh.append(decision)
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




def load_preprocessing_module(project: Path):
    """Load the existing preprocessing module without changing its source."""
    candidates = [
        project / 'scripts' / 'eeg' / 'preprocessing.py',
        project / 'scripts' / 'preprocessing.py',
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            'Cannot find preprocessing.py under scripts/ or scripts/eeg/. '
            'Preprocessing has not been run; audit stopped.'
        )
    spec = importlib.util.spec_from_file_location('eeg_existing_preprocessing', source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import preprocessing script: {source}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'analyze'):
        raise AttributeError(f'{source} does not define analyze()')
    print(f'Using shared preprocessing: {source}', flush=True)
    return module


def run_preprocessing_first(path: Path, project: Path, preprocessor,
                            min_average_rate: float, save_figures: bool = False, args=None):
    """Write the shared two-second outputs once, before audit and human review."""
    sid, phase = preprocessor.recording_identity(path)
    result = {'preprocessing_status':'NOT_RUN', 'preprocessing_error':'',
              'preprocessing_result_path':'', 'preprocessed_path':''}
    folder = project/'outputs'/'eeg'/'bandpower'/('sub_'+sid)
    bundle = project/'data'/'processed'/'eeg'/('sub'+sid)/(path.stem+'_Preprocessed.npz')
    settings = {}
    if args is not None:
        settings = {key:getattr(args,key) for key in (
            'raw_offset_uv','amplitude_limit_uv','gap_factor','flat_std','flat_run_ms','bound_fraction')}
        settings['timing_tolerance'] = args.window_duration_tolerance
    try:
        processed = preprocessor.process_recording(
            path, min_average_rate=min_average_rate,
            manual_dir=project/'outputs'/'eeg'/'manual_annotations'/('sub'+sid), **settings)
        preprocessor.save_preprocessed(processed, bundle)
        _,saved,_ = preprocessor.export_outputs(
            processed,path,folder,save_figure=save_figures,save_spectrogram=True)
        result.update(preprocessing_status='COMPLETED',preprocessing_result_path=str(saved),
                      preprocessed_path=str(bundle),
                      preprocessing_id=processed['metadata']['preprocessing_id'])
    except preprocessor.IneligibleRecording as exc:
        result.update(preprocessing_status='SKIPPED_RATE_OR_LENGTH',preprocessing_error=str(exc))
    except Exception as exc:
        result.update(preprocessing_status='FAILED',
                      preprocessing_error=f'{type(exc).__name__}: {exc}')
    return result

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
    p.add_argument('--raw-offset-uv',type=float,default=725.,help='Muse RAW offset in µV; use 0 for signed data')
    p.add_argument('--amplitude-limit-uv',type=float,default=200.,help='Hard reject outside +/- this µV limit')
    p.add_argument('--ptp-warning',type=float,default=500.,help='Warning only; units not verified')
    p.add_argument('--jump-warning',type=float,default=300.,help='Warning only; units not verified')
    p.add_argument('--outlier-mad',type=float,default=5.)
    p.add_argument('--manual-review',type=Path,default=None,help='Optional reviewer decisions CSV; do not edit original raw data')
    p.add_argument('--skip-review-figures',action='store_true',help='Skip detailed per-recording review galleries')
    p.add_argument(
        '--preprocessing-figures',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Export combined preprocessing TIFFs (default: enabled). '
             'Spectrogram PNGs are always exported for eligible recordings.',
    )
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
    global PREPROCESSOR
    PREPROCESSOR = load_preprocessing_module(project)
    preprocessor = PREPROCESSOR
    infos=[];windows=[];summary=[]
    out.mkdir(parents=True,exist_ok=True)
    for i,path in enumerate(paths,1):
        sub=next((v for v in path.parts if re.fullmatch(r'sub\d+',v)),path.parent.name)
        print(f'[{i}/{len(paths)}] {sub}/{path.name}',flush=True)
        pre_result = run_preprocessing_first(
            path, project, preprocessor, a.min_average_rate,
            save_figures=a.preprocessing_figures, args=a)
        print('  Preprocessing:', pre_result['preprocessing_status'], flush=True)
        if pre_result['preprocessing_error']:
            print('    Reason:', pre_result['preprocessing_error'], flush=True)
        if preprocessor is not None and pre_result['preprocessing_status'] == 'FAILED':
            info = {'subject':sub, 'phase':re.sub(r'^\d+_', '', path.stem).replace('_eeg',''),
                    'recording':path.name, 'source':str(path), 'audit_version':VERSION,
                    'status':'PREPROCESSING_FAILED', 'error':pre_result['preprocessing_error']}
            window_rows, summaries = [], []
        else:
            info,window_rows,summaries=audit_one(path,out/'figures'/sub,a,pre_result)
        info.update(pre_result)
        infos.append(info);windows.extend(window_rows);summary.extend(summaries)
        print('  ->',info['status'],f"({info.get('paired_pass_windows','-')} paired windows)",flush=True)
    pd.DataFrame(infos).to_csv(out/'eeg_recording_qc.csv',index=False)
    pd.DataFrame([{'subject': i.get('subject'), 'phase': i.get('phase'),
                   'recording': i.get('recording'),
                   'preprocessing_status': i.get('preprocessing_status'),
                   'preprocessing_error': i.get('preprocessing_error'),
                   'preprocessing_result_path': i.get('preprocessing_result_path')}
                  for i in infos]).to_csv(out/'eeg_preprocessing_execution_report.csv', index=False)
    window_columns = list(PREPROCESSOR.QC_COLUMNS)+['subject','phase','recording','source',
        'audit_version','preprocessed_path','preprocessing_id','selection','analysis_id',
        'alternate_beta_alpha_ratio','relative_filter_sensitivity']
    pd.DataFrame(windows,columns=window_columns).to_csv(out/'eeg_window_qc_and_power.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'eeg_frontal_band_summary.csv',index=False)
    # Explicit pairing only; no inferential tests on windows as independent observations.
    sums=pd.DataFrame(summary)
    if not sums.empty:
        prim=sums[(sums.method=='paired') & sums.channel.eq('AF7')].copy()
        prim=prim[['subject','phase','recording','accepted_windows']]
        prim.to_csv(out/'eeg_paired_windows_inventory.csv',index=False)
    else:
        pd.DataFrame(columns=['subject','phase','recording','accepted_windows']).to_csv(
            out/'eeg_paired_windows_inventory.csv',index=False)
    settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    (out/'audit_settings.json').write_text(json.dumps({'version':VERSION,'fs_assumed':FS,'window_s':WINDOW_SEC,'step_s':STEP_SEC,'bands_hz':{**BANDS,'theta':list(PREPROCESSOR.BANDS['theta'])},'preprocessor_source':str(Path(PREPROCESSOR.__file__)), 'audit_spectral_method':'preprocessing.py PSD and band integration on notch-filtered QC windows', 'eyes_open_subjects':[39,52], 'eyes_open_rule':'second half of each recording by elapsed time', 'amplitude_limit_uv':a.amplitude_limit_uv, 'raw_offset_uv':a.raw_offset_uv, 'settings':settings},indent=2))
    create_second_pass(out, a, root)
    print('Audit and collected powers use the same accepted 2-s Welch epochs. Manual BAD intervals are applied on rerun.', flush=True)
    print('\nReports saved in:',out)
    print(pd.DataFrame(infos).status.value_counts().to_string())

if __name__=='__main__':main()
