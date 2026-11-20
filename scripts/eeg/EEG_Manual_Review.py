#!/usr/bin/env python3
"""Manually annotate saved, preprocessed AF7/AF8 EEG with MNE.

Run from the Puberty_VR project directory:
  python3 scripts/eeg/EEG_Manual_Review.py --subject 029 --phase pre_vr --channel ALL
  python3 scripts/eeg/EEG_Manual_Review.py --subject 029 --phase pre_vr --channel AF7 --start 195
  python3 scripts/eeg/EEG_Manual_Review.py --subject 029 --phase pre_vr --channel AF7 --list-flags

This viewer does not change the source recording or automatically classify artifacts.
Annotations are stored in MNE FIF format plus a reviewer-friendly CSV, separately
for each recording, displayed channel selection, and continuous time span.

Important: The display assumes RAW values are microvolts. Verify the acquisition
unit definition before interpreting absolute amplitudes or PSD. Original RAW
samples (not the visual scaling) are unchanged.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import mne
import preprocessing
import numpy as np
import pandas as pd

FS = 256.0  # Nominal Muse raw EEG sample rate; do not infer from rounded timestamp dt.
CHANNEL_COLUMNS = {'AF7': 'RAW_AF7', 'AF8': 'RAW_AF8'}
CHANNEL_ORDER = list(CHANNEL_COLUMNS)
LEGACY_TIMELINE_COLUMNS = ('RAW_TP9', 'RAW_TP10')


def project_dir():
    script = Path(__file__).resolve()
    for p in script.parents:
        if (p / 'data' / 'raw' / 'subject_data').is_dir():
            return p
    return Path.cwd()


def spans_for_times(seconds):
    cuts = np.flatnonzero(np.diff(seconds) > 3.5 / FS) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(seconds)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def select_span(spans, times, start):
    if start is not None:
        for i, (a, b) in enumerate(spans):
            if times[a] <= start <= times[b-1]:
                return i
        choices = ', '.join(f'{times[a]:.2f}–{times[b-1]:.2f}' for a,b in spans)
        raise ValueError(f'Requested --start={start} s falls outside available continuous spans: {choices}')
    return max(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])


def print_audit_flags(project, source, channel, audit_windows=None, limit=20):
    if audit_windows is not None:
        audit_path = audit_windows
    else:
        options = [project / 'outputs' / 'eeg' / folder / 'eeg_window_qc_and_power.csv'
                   for folder in ('audit',)]
        audit_path = next((q for q in options if q.exists()), options[-1])
    if not audit_path.exists():
        print(f'No audit window report at: {audit_path}; pass --audit-windows if stored elsewhere')
        return
    df = pd.read_csv(audit_path, low_memory=False)
    f = df[df['recording'].astype(str) == source.name].copy()
    channels = CHANNEL_ORDER if channel in ('ALL', 'BOTH') else [channel]
    f = f[f.channel.isin(channels)]
    if f.empty:
        print(f'No audit window entries for {source.name} / {channel}')
        return
    hard = f['hard_reasons'].fillna('').astype(str).str.strip()
    warning = f['warning_reasons'].fillna('').astype(str).str.strip()
    flagged = f[(hard != '') | (warning != '')].copy()
    flagged['severity'] = np.where(flagged['hard_reasons'].fillna('').astype(str) != '', 'HARD_FLAG', 'WARNING')
    flagged['reason'] = flagged['hard_reasons'].fillna('').astype(str) + ' ' + flagged['warning_reasons'].fillna('').astype(str)
    flagged = flagged.sort_values(['severity','start_s'], ascending=[True, True])
    print(f'\nAudit flags for {source.name}: {len(flagged)} flagged channel-windows')
    if not flagged.empty:
        print(flagged[['channel','start_s','end_s','severity','reason']].head(limit).to_string(index=False))
    print('\nTo view one, use --start <start_s>. These are automated SUSPECTED artifacts.')


def export_annotation_pdf(raw, annotations, output_pdf, source_name, subject, phase,
                          channels, recording_seconds, original_start_index, fs=256):
    """Write one PDF page per annotation, retaining neighboring signal context.

    The displayed signal is the MNE in-memory, saved notch-filtered trace.
    Elapsed recording times are derived from the original recording timestamps.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import numpy as np

    values_uv = raw.get_data() * 1e6
    sample_count = values_uv.shape[1]
    with PdfPages(output_pdf) as pdf:
        if not len(annotations):
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.axis('off')
            ax.text(0.08, 0.8, f'{subject} | {phase} | {source_name}', fontsize=15)
            ax.text(0.08, 0.65, 'No annotations saved for this review.', fontsize=12)
            ax.text(0.08, 0.52, 'This does not certify the recording as artifact-free.', fontsize=11)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return

        for number, (entry, channel_names) in enumerate(zip(annotations, annotations.ch_names), 1):
            onset = float(entry['onset'])
            duration = float(entry['duration'])
            finish = onset + duration
            # At least two seconds of surrounding context where available.
            padding = max(1.5, min(3.0, duration))
            plot_start = max(0.0, onset - padding)
            plot_end = min((sample_count - 1) / fs, finish + padding)
            start_idx = max(0, int(np.floor(plot_start * fs)))
            stop_idx = min(sample_count, int(np.ceil(plot_end * fs)) + 1)
            times = np.arange(start_idx, stop_idx) / fs
            absolute_start_idx = min(len(recording_seconds)-1, original_start_index + min(sample_count-1, int(round(onset * fs))))
            absolute_end_idx = min(len(recording_seconds)-1, original_start_index + min(sample_count-1, int(round(finish * fs))))
            original_begin = float(recording_seconds[absolute_start_idx])
            original_end = float(recording_seconds[absolute_end_idx])
            affected = list(channel_names) if channel_names else ['Global annotation (all loaded channels)']
            fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(12, 2.1 * len(channels) + 2.8), squeeze=False)
            axes = axes.ravel()
            for i, (channel, ax) in enumerate(zip(channels, axes)):
                trace = values_uv[i, start_idx:stop_idx]
                # Center the plot display without changing the stored data.
                # Preserve the same offset-corrected amplitudes as the viewer.
                ax.plot(times, trace, lw=0.75, color='#253545', rasterized=True)
                ax.axvspan(onset, finish, color='#e7ad68', alpha=0.30)
                ax.axvline(onset, color='#aa5925', lw=0.8, ls='--')
                ax.axvline(finish, color='#aa5925', lw=0.8, ls='--')
                ax.set_ylabel(f'{channel}\nµV*', rotation=0, ha='right', va='center', labelpad=35)
                ax.grid(alpha=0.15)
            axes[-1].set_xlabel('Time within selected continuous span (seconds)')
            fig.suptitle(f'{subject} | {phase} | annotation {number}/{len(annotations)}', y=.995, fontsize=13, fontweight='bold')
            fig.text(.10, .955, f'{entry["description"]} | Scope: {", ".join(affected)}', fontsize=10)
            fig.text(.10, .930, f'Selected span: {onset:.3f}–{finish:.3f} s | Original recording elapsed: {original_begin:.3f}–{original_end:.3f} s', fontsize=9)
            fig.text(.10, .906, f'Source: {source_name} | Shading = reviewer-selected interval (suspected, not independently confirmed)', fontsize=8)
            fig.text(.10, .025, '*Amplitude assumes source units are µV; verify hardware scaling. Saved preprocessing shown for review. Automatic QC and human decisions remain explicit.', fontsize=8)
            fig.subplots_adjust(top=.86, bottom=.09, left=.15, right=.97, hspace=.22)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='029')
    p.add_argument('--phase', default='pre_vr')
    p.add_argument('--input', type=Path, help='Saved _Preprocessed.npz, or its original CSV/ZIP path')
    p.add_argument('--channel', choices=['ALL','BOTH','AF7','AF8'], default='ALL')
    p.add_argument('--start', type=float, help='Original recording elapsed seconds')
    p.add_argument('--duration', type=float, default=10.)
    p.add_argument('--scale-uv', type=float, default=200.)
    p.add_argument('--list-flags', action='store_true')
    p.add_argument('--audit-windows', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    project = project_dir()
    if args.input and args.input.suffix.lower() == '.npz':
        bundle = args.input.expanduser().resolve()
    else:
        stem = args.input.stem if args.input else f'{args.subject.zfill(3)}_{args.phase}_eeg'
        sid, _ = preprocessing.recording_identity(Path(stem+'.csv'))
        bundle = project/'data'/'processed'/'eeg'/('sub'+sid)/(stem+'_Preprocessed.npz')
    if not bundle.is_file():
        p.error(f'Preprocessed EEG not found: {bundle}. Run EEG_Audit.py first.')
    data = preprocessing.load_preprocessed(bundle)
    meta = data['metadata']
    sid, phase = meta['subject_id'], meta['phase']
    source = Path(meta['recording'])
    seconds = data['times']
    channels = list(preprocessing.CHANNELS) if args.channel in ('ALL','BOTH') else [args.channel]
    if args.list_flags:
        print_audit_flags(project, source, args.channel, args.audit_windows)
        return
    spans = [tuple(map(int, pair)) for pair in data['spans']
             if pair[1]-pair[0] >= preprocessing.WINDOW_LENGTH]
    if not spans:
        p.error('No complete preprocessed span available to review.')
    span_idx = select_span(spans, seconds, args.start)
    a, b = spans[span_idx]
    values = data['filtered_uv'][a:b, [preprocessing.CHANNELS.index(c) for c in channels]].T
    clean = values.copy()
    preannotations = []
    for ci, channel in enumerate(channels):
        invalid = ~np.isfinite(clean[ci])
        edges = np.flatnonzero(np.diff(np.r_[False, invalid, False].astype(int)))
        for left, right in edges.reshape(-1, 2):
            preannotations.append((left/FS, (right-left)/FS, 'BAD_auto_missing', (channel,)))
        clean[ci] = pd.Series(clean[ci]).interpolate(limit_direction='both').fillna(0.).to_numpy()
    qc = data['windows']
    qc['automatic_pass'] = qc.automatic_pass.astype(str).str.lower().eq('true')
    for channel in channels:
        selected = qc[(qc.channel == channel) & ~qc.automatic_pass
                      & (qc.window_index >= a) & (qc.stop_index <= b)].sort_values('window_index')
        ranges = []
        for row in selected.itertuples():
            if ranges and row.window_index <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], int(row.stop_index))
            else:
                ranges.append([int(row.window_index), int(row.stop_index)])
        for left, right in ranges:
            preannotations.append(((left-a)/FS, (right-left)/FS,
                                   'BAD_auto_rejected_window', (channel,)))
    info = mne.create_info(ch_names=channels, sfreq=FS, ch_types=['eeg']*len(channels))
    raw = mne.io.RawArray(clean*1e-6, info, verbose=False)
    output = args.output_dir or project/'outputs'/'eeg'/'manual_annotations'/('sub'+sid)
    suffix = 'AF7_AF8' if len(channels) == 2 else channels[0]
    stem = f"{meta['recording_stem']}_{suffix}_processed_{meta['preprocessing_id'][:12]}_span{span_idx+1}"
    fif_path = output/(stem+'-annot.fif')
    csv_path = output/(stem+'_intervals.csv')
    annotations = mne.Annotations(onset=[], duration=[], description=[])
    if fif_path.is_file():
        previous = mne.read_annotations(fif_path)
        for entry, scope in zip(previous, previous.ch_names):
            if not str(entry['description']).startswith('BAD_auto_'):
                annotations.append(entry['onset'], entry['duration'], entry['description'], ch_names=[scope])
    for onset, duration, label, scope in preannotations:
        annotations.append(onset, duration, label, ch_names=[scope])
    raw.set_annotations(annotations)
    print(f"Reviewing saved preprocessing {meta['preprocessing_id'][:12]} | "
          f"{meta['selection']} | {seconds[a]:.3f}–{seconds[b-1]+1/FS:.3f} original seconds")
    print('Press A and add BAD_blink, BAD_muscle or REVIEW_uncertain intervals.')
    print('BAD_auto_ spans show automatic exclusions. Review each continuous span separately.')
    if args.dry_run:
        print('Dry run successful. Source:', bundle)
        return
    output.mkdir(parents=True,exist_ok=True)
    mne.viz.set_browser_backend('matplotlib')
    duration = min(max(args.duration, 2.), raw.times[-1])
    start = max(0., (np.searchsorted(seconds[a:b], args.start)/FS)) if args.start is not None else 0.
    raw.plot(start=min(start,max(0.,raw.times[-1]-duration)),duration=duration,
             n_channels=len(channels),scalings={'eeg':args.scale_uv*1e-6},
             remove_dc=False,block=True,title=source.name+' | preprocessed | '+suffix)
    reviewed = raw.annotations
    if len(channels) == 1 and len(reviewed):
        reviewed = mne.Annotations(onset=reviewed.onset,duration=reviewed.duration,
            description=reviewed.description,orig_time=reviewed.orig_time,
            ch_names=[scope if scope else (channels[0],) for scope in reviewed.ch_names])
    rows = []
    for entry, scope in zip(reviewed, reviewed.ch_names):
        left = max(0,min(b-a-1,int(np.floor(entry['onset']*FS+1e-8))))
        stop = max(left+1,min(b-a,int(np.ceil((entry['onset']+entry['duration'])*FS-1e-8))))
        rows.append({
            'subject':'sub'+sid,'recording':meta['recording'],'channel_scope':args.channel,
            'span_number':span_idx+1,'label':entry['description'],
            'affected_channels':'|'.join(scope if scope else channels),
            'display_start_s':float(entry['onset']),
            'display_end_s':float(entry['onset']+entry['duration']),
            'recording_start_s':float(seconds[a+left]),
            'recording_end_s':float(seconds[a+stop-1]+1/FS),
            'recording_start_timestamp':str(data['timestamps'][a+left]),
            'recording_end_timestamp':str(data['timestamps'][a+stop-1]),
            'duration_s':float(entry['duration']),'reviewer_note':'',
            'preprocessing_id':meta['preprocessing_id'],'pipeline_version':preprocessing.VERSION})
    columns = ['subject','recording','channel_scope','span_number','label','affected_channels',
        'display_start_s','display_end_s','recording_start_s','recording_end_s',
        'recording_start_timestamp','recording_end_timestamp','duration_s','reviewer_note',
        'preprocessing_id','pipeline_version']
    table = pd.DataFrame(rows,columns=columns)
    if csv_path.exists() and len(table):
        previous = pd.read_csv(csv_path,keep_default_na=False)
        notes = {(row.label,float(row.display_start_s),float(row.display_end_s)):row.reviewer_note
                 for row in previous.itertuples()}
        table['reviewer_note'] = [notes.get((row.label,row.display_start_s,row.display_end_s),'')
                                  for row in table.itertuples()]
    reviewed.save(fif_path,overwrite=True)
    table.to_csv(csv_path,index=False)
    pdf = output/(stem+'_annotated_traces.pdf')
    export_annotation_pdf(raw,reviewed,pdf,source.name,'sub'+sid,phase,channels,seconds,a,fs=FS)
    print('Saved manual intervals:',csv_path)
    print('Rerun EEG_Audit.py then process_all_eeg.py to apply matching BAD_ intervals.')


if __name__ == '__main__':
    main()
