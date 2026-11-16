"""Puberty & VR: raw Polar HR extraction, plots and repeated-measures statistics.

Install: pip install numpy pandas matplotlib scipy statsmodels
Place this file in PROJECT_ROOT/scripts/heart_rate/ (or scripts/).
Run: python scripts/heart_rate/hr_analysis.py
Optional: --raw-dir /path/to/subject_data --metadata /path/to/participants.csv
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Locate the shared path module whether this script is directly in scripts/
# or in a subdirectory such as scripts/heart_rate/.
for parent in Path(__file__).resolve().parents:
    if (parent / "project_paths.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise SystemExit("Could not find project_paths.py in this script's parent directories")

from project_paths import (RAW_SUBJECT_DIR, PREVIOUS_SUBJECTS_DIR,
    HEART_RATE_PROCESSED_DIR, HEART_RATE_ANALYSIS_DATA_DIR,
    HEART_RATE_OUTPUTS_DIR, METADATA_DIR)

PHASES = ['Baseline','preVR','VRprep','VRSpeech','VRMath','postVR','Survey']
KEYS = ['baseline','pre_vr','prep','speech','math','post_vr','recovery']
# Note: raw *recovery_hr* corresponds to the T7 survey recording by convention; verify experimentally.
MATCH = {'baseline':'Baseline','pre_vr':'preVR','prep':'VRprep','speech':'VRSpeech','math':'VRMath','post_vr':'postVR','recovery':'Survey'}
GOOD = {'010','014','015','018','019','020','022','025'}
POOR = {'005','007','012','013','016','023','024'}


def subject_id(folder):
    m = re.search(r'(?:sub(?:ject)?[_-]?)?(\d{3})$', folder.name, re.I)
    return m.group(1) if m else None


def parse_hr(path):
    """Polar CSV: metadata row, values row, sample header, then 1 Hz samples."""
    with path.open(encoding='utf-8-sig', errors='replace', newline='') as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 4:
        raise ValueError('No sample rows')
    header = rows[0]
    device = dict(zip(header, rows[1]))
    sample_header = rows[2]
    if 'HR (bpm)' not in sample_header:
        raise ValueError('Missing HR (bpm) sample column')
    i = sample_header.index('HR (bpm)')
    values = pd.to_numeric(pd.Series([r[i] if len(r)>i else '' for r in rows[3:]]), errors='coerce')
    valid = values.dropna()
    if valid.empty:
        raise ValueError('No valid HR samples')
    reported = pd.to_numeric(device.get('Average heart rate (bpm)'), errors='coerce')
    return {'mean_hr':float(valid.mean()), 'reported_mean_hr':float(reported) if pd.notna(reported) else np.nan,
            'n_samples':int(len(valid)), 'n_missing_samples':int(values.isna().sum()),
            'min_hr':float(valid.min()),'max_hr':float(valid.max()),'sd_hr':float(valid.std()) if len(valid)>1 else np.nan}


def locate_files(root):
    result = {}
    if not root.exists(): return result
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() != '.csv': continue
        if any(part.startswith('.') or part == '__MACOSX' for part in path.parts): continue
        match = re.search(r'(\d{3})_(baseline|pre_vr|prep|speech|math|post_vr|recovery)_hr\.csv$',path.name,re.I)
        if not match: continue
        sid, kind = match.group(1), match.group(2).lower()
        # Guard against mismatched parent IDs when a recognizable subject folder exists
        parent_ids = [subject_id(p) for p in path.parents if subject_id(p)]
        if parent_ids and sid not in parent_ids:
            warnings.warn(f'Skipping mismatched subject path: {path}')
            continue
        key=(sid, MATCH[kind])
        if key in result:
            warnings.warn(f'Duplicate HR file for {key}: retaining {result[key]}, ignoring {path}')
        else: result[key]=path
    return result


def extract(root):
    files=locate_files(root)
    records=[]
    for (sid, phase),path in sorted(files.items()):
        try:
            d=parse_hr(path)
            records.append({'subject':sid,'phase':phase,'timepoint':PHASES.index(phase)+1,'source_file':str(path),**d})
        except (ValueError, OSError) as e:
            warnings.warn(f'Skipping {path}: {e}')
    return pd.DataFrame(records, columns=['subject','phase','timepoint','source_file','mean_hr','reported_mean_hr','n_samples','n_missing_samples','min_hr','max_hr','sd_hr'])


def metadata_table(path):
    if path is None or not path.exists():
        return pd.DataFrame(columns=['subject','age','learner_group'])
    d=pd.read_csv(path,dtype={'subject':str})
    if 'subject' not in d: raise ValueError('Metadata must contain subject column')
    d['subject']=d['subject'].astype(str).str.extract(r'(\d+)',expand=False).str.zfill(3)
    if d.subject.duplicated().any(): raise ValueError('Duplicate subjects in metadata')
    if 'age' not in d: d['age']=np.nan
    d['age']=pd.to_numeric(d['age'],errors='coerce')
    if 'learner_group' not in d: d['learner_group']=pd.NA
    return d[['subject','age','learner_group']]


def summary(long):
    return long.groupby('phase',observed=True)['mean_hr'].agg(n='count',mean='mean',sd='std').reindex(PHASES).assign(se=lambda d:d.sd/np.sqrt(d.n))


def save_plot(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)


def plot_individual(wide,output,title,average=True,only_average=False):
    fig,ax=plt.subplots(figsize=(11,6));x=np.arange(7)
    if not only_average:
        for sid,row in wide.iterrows():
            ax.plot(x,row.reindex(PHASES).to_numpy(dtype=float),marker='o',ms=3,lw=1,alpha=.55,label=str(sid))
    if average and not wide.empty:
        s=summary(wide.stack().rename('mean_hr').reset_index().rename(columns={'level_0':'subject','level_1':'phase'}))
        ax.errorbar(x,s['mean'],yerr=s['se'],marker='o',lw=2.4,color='black',capsize=4,label='Average ± SE')
    ax.set_xticks(x,PHASES,rotation=25,ha='right');ax.set_ylabel('Mean heart rate (bpm)');ax.set_title(title);ax.grid(alpha=.25)
    if not only_average and not wide.empty and len(wide)<=20: ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    elif only_average: ax.legend()
    save_plot(fig,output)


def compare_groups(wide,groups,labels,path,title):
    fig,ax=plt.subplots(figsize=(11,6));x=np.arange(7); any_group=False
    for label in labels:
        sub=wide.loc[groups.reindex(wide.index).eq(label).fillna(False)]
        if sub.empty: continue
        any_group=True
        means=sub.mean(axis=0).reindex(PHASES)
        se=sub.std(axis=0).reindex(PHASES)/np.sqrt(sub.count(axis=0).reindex(PHASES))
        ax.errorbar(x,means,yerr=se,marker='o',capsize=3,label=f'{label} (n={len(sub)})')
    ax.set_xticks(x,PHASES,rotation=25,ha='right');ax.set_ylabel('Mean heart rate (bpm)');ax.set_title(title);ax.grid(alpha=.25)
    if any_group:ax.legend()
    else:ax.text(.5,.5,'No group metadata available',transform=ax.transAxes,ha='center')
    save_plot(fig,path)


def stats_analysis(wide,folder):
    """Complete-case within-subject ANOVA; pairwise available-case and complete-case tests."""
    from statsmodels.stats.anova import AnovaRM
    folder.mkdir(parents=True,exist_ok=True)
    complete=wide.dropna(subset=PHASES)
    result={'complete_subjects':len(complete),'total_subjects':len(wide),'status':''}
    if len(complete)>=3:
        d=complete[PHASES].stack().rename('HR').reset_index()
        d.columns=['subject','phase','HR']
        try:
            fit=AnovaRM(d,depvar='HR',subject='subject',within=['phase']).fit()
            table=fit.anova_table.reset_index().rename(columns={'index':'effect'})
            table.to_csv(folder/'HR_ANOVA_complete_case.csv',index=False)
            result['status']='ANOVA complete; sphericity not assessed (interpret uncorrected p cautiously)'
        except Exception as e: result['status']=f'ANOVA failed: {e}'
    else:result['status']='ANOVA not run: fewer than 3 complete subjects'
    for label,data in [('available_case',wide),('complete_case',complete)]:
        pairs=[]
        for i,a in enumerate(PHASES):
            for b in PHASES[i+1:]:
                pair=data[[a,b]].dropna();n=len(pair)
                if n>=2:
                    delta=pair[b]-pair[a]
                    t,p=stats.ttest_rel(pair[b],pair[a]); mean_delta=float(delta.mean())
                    if not np.isfinite(p):p=np.nan
                else:t=p=mean_delta=np.nan
                pairs.append({'phase_a':a,'phase_b':b,'n':n,'mean_change_b_minus_a':mean_delta,'t':t,'p':p})
        df=pd.DataFrame(pairs)
        # Holm adjustment over all finite pairwise tests
        valid=df.p.notna();pvals=df.loc[valid,'p'].to_numpy()
        if len(pvals):
            order=np.argsort(pvals);m=len(pvals);adjusted=np.empty(m);running=0
            for rank,idx in enumerate(order):
                running=max(running,(m-rank)*pvals[idx]);adjusted[idx]=min(1.,running)
            df.loc[valid,'p_holm']=adjusted
        df.to_csv(folder/f'HR_pairwise_{label}.csv',index=False)
    pd.DataFrame([result]).to_csv(folder/'HR_statistics_status.csv',index=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=RAW_SUBJECT_DIR)
    p.add_argument('--metadata',type=Path,default=METADATA_DIR/'hr_participants.csv')
    p.add_argument('--no-stats',action='store_true')
    a=p.parse_args()
    for folder in (HEART_RATE_PROCESSED_DIR,HEART_RATE_ANALYSIS_DATA_DIR,HEART_RATE_OUTPUTS_DIR):folder.mkdir(parents=True,exist_ok=True)
    plot_dir=HEART_RATE_OUTPUTS_DIR/'plots';stat_dir=HEART_RATE_OUTPUTS_DIR/'statistics'
    plot_dir.mkdir(parents=True,exist_ok=True)
    long=extract(a.raw_dir)
    if long.empty:raise SystemExit(f'No raw HR CSVs found under {a.raw_dir}; use --raw-dir if needed')
    long.to_csv(HEART_RATE_PROCESSED_DIR/'HR_recordings.csv',index=False)
    wide=long.pivot(index='subject',columns='phase',values='mean_hr').reindex(columns=PHASES).sort_index()
    meta=metadata_table(a.metadata).set_index('subject')
    wide.to_csv(HEART_RATE_ANALYSIS_DATA_DIR/'HR_Individual.csv',index_label='subject')
    # Legacy-compatible alias, with age column (if supplied)
    updated=wide.join(meta[['age']],how='left') if not meta.empty else wide.assign(age=np.nan)
    updated.to_csv(HEART_RATE_ANALYSIS_DATA_DIR/'HR_update.csv',index_label='subject')
    summary(long).to_csv(HEART_RATE_ANALYSIS_DATA_DIR/'HR_Summary.csv',index_label='phase')
    plot_individual(wide,plot_dir/'HR_Individuals.png','Individual heart rate across VR phases',average=False)
    plot_individual(wide,plot_dir/'HR_Average_Only.png','Average heart rate across VR phases',only_average=True)
    plot_individual(wide,plot_dir/'HR_Combined.png','Individual and average heart rate')
    age=meta.age.reindex(wide.index) if 'age' in meta else pd.Series(np.nan,index=wide.index)
    agegroup=pd.Series(np.where(age.notna(),np.where(age<18,'Minor','Adult'),None),index=wide.index)
    for label,filename in [('Minor','HR_Minors.png'),('Adult','HR_Adults.png')]:
        plot_individual(wide.loc[agegroup==label],plot_dir/filename,f'{label} participants: heart rate')
    compare_groups(wide,agegroup,['Minor','Adult'],plot_dir/'HR_Age_Average_Diff.png','Mean heart rate by age group (± SE)')
    group=pd.Series({sid:('Good' if sid in GOOD else 'Poor' if sid in POOR else None) for sid in wide.index})
    if 'learner_group' in meta:
        group.update(meta.learner_group.dropna().astype(str).str.title())
    for label,filename in [('Good','HR_Good_Learners.png'),('Poor','HR_Poor_Learners.png')]:
        plot_individual(wide.loc[group.reindex(wide.index)==label],plot_dir/filename,f'{label} learners: heart rate')
    compare_groups(wide,group,['Good','Poor'],plot_dir/'HR_Learner_Average_Diff.png','Mean heart rate by learning group (± SE)')
    pd.DataFrame({'subject':wide.index,'age':age.values,'age_group':agegroup.values,'learner_group':group.reindex(wide.index).values}).to_csv(HEART_RATE_ANALYSIS_DATA_DIR/'HR_groups.csv',index=False)
    if not a.no_stats:print(stats_analysis(wide,stat_dir))
    print(f'Processed {len(wide)} subjects and {len(long)} recordings. Outputs: {HEART_RATE_OUTPUTS_DIR}')
    if age.isna().all():print('Age plots empty: provide data/metadata/hr_participants.csv with subject,age columns.')

if __name__=='__main__':main()
