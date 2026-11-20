#!/usr/bin/env python3
"""Aggregate EXISTING outputs of preprocessing.py; never process EEG.

Run AFTER EEG_Audit.py (which calls preprocessing.py first).
The preprocessor is the sole source of spectral filtering and PSD routines.
This script reads the already-exported _FrontalLogPower.csv tables, checks
preprocessing/audit status when available, and creates cohort-level tables.
It does not read or modify raw signal samples or recompute power.
"""
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

PHASE_TO_TIMEPOINT = {
    'baseline': 'baseline',
    'pre_vr': 'anticipation',
    'post_vr': 'acute_stress_response',
    'recovery': 'recovery',
}
CHANNELS = ('AF7', 'AF8')
BANDS = ('theta', 'alpha', 'beta')
PIPELINE_VERSION = '4.2-eyes-open-welch2s-original-layout'


def collect(project: Path, *, include_unverified: bool = False) -> int:
    output_root = project / 'outputs' / 'eeg'
    audit_file = output_root / 'audit' / 'eeg_recording_qc.csv'
    band_root = output_root / 'bandpower'
    target = project / 'data' / 'analysis_data' / 'eeg'
    # Honor installed project's existing analysis-data path when available.
    target.mkdir(parents=True, exist_ok=True)
    if not band_root.is_dir() and not audit_file.is_file():
        raise FileNotFoundError(f'No preprocessing outputs or audit found: {band_root}. Run EEG_Audit.py first.')

    if audit_file.exists():
        audit = pd.read_csv(audit_file, dtype={'subject': str}).fillna('')
        audit['sid'] = audit['subject'].astype(str).str.extract(r'(\d+)', expand=False).str.zfill(3)
        audit_lookup = {(r.sid, str(r.phase)): r for r in audit.itertuples(index=False)}
    else:
        audit_lookup = {}
        print('WARNING: audit report not found; results will be marked QC_UNVERIFIED.')

    report = []
    long = []
    for folder in sorted(band_root.glob('sub_*')):
        if not folder.is_dir():
            continue
        m = re.search(r'(\d+)$', folder.name)
        if m is None:
            continue
        sid = m.group(1).zfill(3)
        for path in sorted(folder.glob('*_FrontalLogPower.csv')):
            phase = re.sub(r'^\d+_', '', path.name.removesuffix('_FrontalLogPower.csv')).removesuffix('_eeg')
            if phase not in PHASE_TO_TIMEPOINT:
                continue
            audit_rec = audit_lookup.get((sid, phase))
            qc_status = str(getattr(audit_rec, 'status', 'QC_UNVERIFIED'))
            processing_status = str(getattr(audit_rec, 'preprocessing_status', 'UNKNOWN'))
            # Reject stale preprocessing exports when the latest audit did not
            # confirm that preprocessing completed for this recording.
            if audit_lookup and audit_rec is None:
                continue
            if not include_unverified and processing_status != 'COMPLETED':
                continue
            row = {'subject_id': sid, 'phase': phase,
                   'timepoint': PHASE_TO_TIMEPOINT[phase],
                   'file_path': str(path.relative_to(project)),
                   'qc_status': qc_status,
                   'processing_status': processing_status}
            try:
                data = pd.read_csv(path)
                if 'pipeline_version' not in data or not data.pipeline_version.eq(PIPELINE_VERSION).all():
                    raise ValueError('Outdated EEG export. Rerun EEG_Audit.py with the updated scripts.')
                metadata_path = path.with_name(path.name.replace('_FrontalLogPower.csv','_Processing.json'))
                metadata = json.loads(metadata_path.read_text())
                if metadata['version'] != PIPELINE_VERSION or not data.preprocessing_id.eq(metadata['preprocessing_id']).all():
                    raise ValueError('Mismatched EEG preprocessing metadata')
                if not data.analysis_id.eq(metadata['analysis_id']).all():
                    raise ValueError('Band table has a different manual exclusion set')
                if audit_rec is not None and str(getattr(audit_rec,'analysis_id','')) != metadata['analysis_id']:
                    raise ValueError('Export does not match the latest audit manual exclusion set')
                if audit_rec is not None and str(getattr(audit_rec,'preprocessing_id','')) != metadata['preprocessing_id']:
                    raise ValueError('Export does not match the latest audit preprocessing ID')
                required = {'channel', 'band', 'power', 'log10_power', 'n_windows'}
                if not required.issubset(data.columns):
                    raise ValueError(f'Missing columns: {sorted(required - set(data.columns))}')
                if len(data) != 6 or set(zip(data.channel, data.band)) != {
                    (c, b) for c in CHANNELS for b in BANDS
                }:
                    raise ValueError('Expected 6 rows: AF7/AF8 x theta/alpha/beta')
                for v in data.itertuples(index=False):
                    power, logpower = float(v.power), float(v.log10_power)
                    if int(v.n_windows) == 0:
                        if not np.isnan(power) or not np.isnan(logpower):
                            raise ValueError('Zero accepted windows must have missing power')
                    elif not np.isfinite(power) or power < 0 or (power > 0 and not np.isclose(np.log10(power),logpower)):
                        raise ValueError(f'Invalid {v.channel} {v.band} power')
                    key = f'{v.channel.lower()}_{v.band}_log10_power'
                    row[key] = logpower
                    long.append({**{k: row[k] for k in ('subject_id','phase','timepoint','file_path',
                                                           'qc_status','processing_status')},
                                 'channel':v.channel, 'band':v.band,
                                 'power': power, 'log10_power': logpower,
                                 'n_windows': int(v.n_windows),
                                 'measurement_timing': 'after_vr_task' if phase == 'post_vr' else phase,
                                  'spectral_source': 'fixed 2-s Welch; automatic and matching manual BAD exclusions',
                                 'pipeline_version': PIPELINE_VERSION,
                                 'preprocessing_id': metadata['preprocessing_id'],
                                 'analysis_id': metadata['analysis_id'],
                                 'selection': metadata['selection'],
                                 'analysis_start_s': metadata['analysis_start_s'],
                                 'analysis_end_s': metadata['analysis_end_s'],
                                 'psd_units': 'uV^2/Hz' })
                counts = data.groupby('channel').n_windows.first()
                row['summary_status'] = ('COLLECTED' if counts.gt(0).all()
                                         else 'COLLECTED_PARTIAL' if counts.gt(0).any()
                                         else 'NO_ACCEPTED_WINDOWS')
                row['error'] = ''
            except Exception as exc:
                row['summary_status'] = 'ERROR'
                row['error'] = str(exc)
                long = [v for v in long if not (v['subject_id'] == sid and v['phase'] == phase)]
            report.append(row)

    if not report:
        print('No eligible preprocessing exports. Writing empty aggregate tables.')
    report_df = pd.DataFrame(report,columns=['subject_id','phase','timepoint','file_path',
                                            'qc_status','processing_status',
                                            'summary_status','error']) if not report else pd.DataFrame(report)
    long_columns = ['subject_id','phase','timepoint','file_path','qc_status',
                    'processing_status','channel','band','power','log10_power','n_windows',
                    'measurement_timing','spectral_source','pipeline_version','preprocessing_id',
                    'analysis_id','selection','analysis_start_s','analysis_end_s','psd_units']
    long_df = pd.DataFrame(long,columns=long_columns)
    columns = ['subject_id','phase','timepoint','file_path','qc_status','processing_status',
               'summary_status','error'] + [f'{ch.lower()}_{band}_log10_power'
                                          for ch in CHANNELS for band in BANDS]
    report_df.reindex(columns=columns).to_csv(target / 'eeg_batch_processing_report.csv', index=False)
    long_df.to_csv(target / 'eeg_frontal_log_bandpower_long.csv', index=False)

    wide_cols = ['subject_id'] + [f'{tp}_{ch.lower()}_{band}_log10_power'
                                  for tp in PHASE_TO_TIMEPOINT.values()
                                  for ch in CHANNELS for band in BANDS]
    if long_df.empty:
        wide = pd.DataFrame(columns=wide_cols)
    else:
        wide = long_df.pivot(index='subject_id', columns=['timepoint','channel','band'],
                             values='log10_power')
        wide.columns = [f'{tp}_{ch.lower()}_{b}_log10_power' for tp,ch,b in wide.columns]
        wide = wide.reset_index().reindex(columns=wide_cols)
    wide.to_csv(target / 'eeg_frontal_log_bandpower_wide.csv', index=False)
    print(f'Collected {len(long_df)} spectral measurements from '
          f'{(report_df.summary_status == "COLLECTED").sum()} recordings.')
    print(f'Reports written to: {target}')
    print('Powers use selected eyes-open periods, 2-s epochs, +/-200 µV QC and matching manual BAD exclusions.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path,
                        default=next((parent for parent in Path(__file__).resolve().parents
                                      if (parent / 'data' / 'raw' / 'subject_data').is_dir()),
                                     Path(__file__).resolve().parents[2]),
                        help='Root Puberty_VR folder')
    parser.add_argument('--include-unverified', action='store_true',
                        help='Include exports without a confirmed COMPLETED preprocessing status')
    args = parser.parse_args()
    return collect(args.project.resolve(), include_unverified=args.include_unverified)


if __name__ == '__main__':
    raise SystemExit(main())
