#!/usr/bin/env python3
"""Manually inspect all four Muse/Mind Monitor raw EEG channels with MNE.

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
import io
import re
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

FS = 256.0  # Nominal Muse raw EEG sample rate; do not infer from rounded timestamp dt.
CHANNEL_COLUMNS = {'TP9': 'RAW_TP9', 'AF7': 'RAW_AF7', 'AF8': 'RAW_AF8', 'TP10': 'RAW_TP10'}
CHANNEL_ORDER = list(CHANNEL_COLUMNS)
TARGET_CHANNELS = ('AF7', 'AF8')


def project_dir():
    script = Path(__file__).resolve()
    for p in script.parents:
        if (p / 'data' / 'raw' / 'subject_data').is_dir():
            return p
    return Path.cwd()


def get_source(args, project):
    if args.input is not None:
        path = args.input.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Input not found: {path}')
        return path
    sub = f'sub{args.subject.zfill(3)}'
    root = project / 'data' / 'raw' / 'subject_data' / sub
    if not root.exists():
        raise FileNotFoundError(f'Subject folder not found: {root}')
    desired = f'{args.subject.zfill(3)}_{args.phase}_eeg'
    files = [p for p in root.iterdir() if p.is_file()
             and p.suffix.lower() in ('.csv', '.zip')
             and p.stem.lower() == desired.lower()]
    if len(files) != 1:
        raise FileNotFoundError(f'Expected one {desired}.csv/.zip in {root}, found {len(files)}')
    return files[0]


def load_csv(source):
    if source.suffix.lower() == '.csv':
        return pd.read_csv(source, low_memory=False)
    if source.suffix.lower() != '.zip':
        raise ValueError(f'Expected .csv or .zip, received {source}')
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist()
                 if n.lower().endswith('.csv')
                 and not n.startswith('__MACOSX/')
                 and not Path(n).name.startswith('._')]
        if len(names) != 1:
            raise ValueError(f'{source.name}: expected one CSV inside ZIP, found {len(names)}')
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle, low_memory=False)


def load_full_rate(source):
    df = load_csv(source)
    required = ['TimeStamp'] + list(CHANNEL_COLUMNS.values())
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    values = df[list(CHANNEL_COLUMNS.values())].apply(pd.to_numeric, errors='coerce')
    # Some Mind Monitor rows carry auxiliary signals rather than raw EEG.
    mask = values.notna().any(axis=1)
    values = values.loc[mask].reset_index(drop=True)
    times = pd.to_datetime(df.loc[mask, 'TimeStamp'], errors='coerce').reset_index(drop=True)
    valid = times.notna().to_numpy()
    values = values.loc[valid].reset_index(drop=True)
    times = times.loc[valid].reset_index(drop=True)
    if len(times) < 1024:
        raise ValueError('Fewer than 1024 timestamped RAW EEG samples (not a useful full-rate recording)')

    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    good = np.r_[True, np.diff(seconds) > 0]
    removed = int((~good).sum())
    seconds = seconds[good]
    times = times.loc[good].reset_index(drop=True)
    values = values.loc[good].reset_index(drop=True)
    seconds -= seconds[0]

    observed_rate = (len(seconds) - 1) / (seconds[-1] - seconds[0])
    positive_dt = np.diff(seconds)
    median_rate = 1 / np.median(positive_dt)
    if not (230 <= median_rate <= 290 and observed_rate >= 180):
        raise ValueError(f'Not confirmed full-rate: median {median_rate:.1f} Hz, average {observed_rate:.1f} Hz')
    return seconds, times, values, removed, median_rate, observed_rate


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


def fill_for_display(samples):
    """Only display interpolation; annotate every missing run as BAD_missing_raw."""
    arr = np.asarray(samples, dtype=float)
    annotations = []
    for chan, col in enumerate(arr):
        missing = ~np.isfinite(col)
        if not missing.any():
            continue
        if np.isfinite(col).sum() < 2:
            raise ValueError(f'{CHANNEL_ORDER[chan]} has insufficient values in selected span')
        starts = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == -1)
        for s, e in zip(starts, ends):
            annotations.append((max(0, (s - 1) / FS), (e - s + 2) / FS,
                                'BAD_missing_raw_sample'))
        arr[chan] = pd.Series(col).interpolate(limit_direction='both').to_numpy()
    return arr, annotations


def print_audit_flags(project, source, channel, audit_windows=None, limit=20):
    if audit_windows is not None:
        audit_path = audit_windows
    else:
        options = [project / 'outputs' / 'eeg' / folder / 'eeg_window_qc_and_power.csv'
                   for folder in ('audit_v2', 'audit')]
        audit_path = next((q for q in options if q.exists()), options[-1])
    if not audit_path.exists():
        print(f'No audit window report at: {audit_path}; pass --audit-windows if stored elsewhere')
        return
    df = pd.read_csv(audit_path, low_memory=False)
    f = df[df['recording'].astype(str) == source.name].copy()
    if channel not in ('ALL', 'BOTH'):
        f = f[f.channel == channel]
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='029', help='e.g. 029')
    p.add_argument('--phase', default='pre_vr', choices=['baseline','pre_vr','post_vr','recovery'])
    p.add_argument('--input', type=Path, help='Optional exact input CSV/ZIP path; overrides subject/phase lookup')
    p.add_argument('--channel', choices=['ALL', 'BOTH', 'TP9', 'AF7', 'AF8', 'TP10'], default='ALL',
                   help='ALL (default) shows four channels; BOTH is a legacy alias for ALL; individual channels can be reviewed separately')
    p.add_argument('--start', type=float, default=None, help='Elapsed seconds since first valid RAW sample (audit coordinates)')
    p.add_argument('--duration', type=float, default=1.5, help='Visible time span in MNE browser, seconds')
    p.add_argument('--list-flags', action='store_true', help='Print representative audit flags and exit')
    p.add_argument('--audit-windows', type=Path, default=None, help='Path to EEG Audit v2 eeg_window_qc_and_power.csv')
    p.add_argument('--dry-run', action='store_true', help='Validate loading and paths without opening GUI')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--scale-uv', type=float, default=800., help='Display scaling in assumed microvolts (default 1600; original 250 made saturated traces illegible). Try 600 for AF7/AF8-only views.')
    args = p.parse_args()

    project = project_dir()
    source = get_source(args, project)
    print('Reading original:', source)
    seconds, timestamps, values, removed, median_rate, observed_rate = load_full_rate(source)
    print(f'{len(seconds):,} raw EEG rows; median dt ~ {median_rate:.1f} Hz; observed ~ {observed_rate:.1f} Hz; dropped non-monotonic timestamp rows: {removed}')
    print('Channel sample-range diagnostics (exact boundary hits are SUSPECTED clipping, not confirmed):')
    for ch in CHANNEL_ORDER:
        vals = pd.to_numeric(values[CHANNEL_COLUMNS[ch]], errors='coerce').to_numpy()
        valid = vals[np.isfinite(vals)]
        if len(valid):
            rail_pct = 100 * np.mean((valid == 0) | (valid == 1450))
            print(f'  {ch:4s}: median={np.median(valid):8.1f}, p01={np.percentile(valid,1):8.1f}, p99={np.percentile(valid,99):8.1f}, exact 0/1450={rail_pct:5.1f}%')
    print('Note: these native values include ~725 offset. Plot is scaled for review; source data unchanged.')
    spans = spans_for_times(seconds)
    span_idx = select_span(spans, seconds, args.start)
    a, b = spans[span_idx]
    segment_seconds = seconds[a:b]
    print(f'Using continuous span {span_idx+1}/{len(spans)}: {seconds[a]:.3f}–{seconds[b-1]:.3f} s ({(b-a)/FS:.1f} nominal seconds)')

    if args.list_flags:
        print_audit_flags(project, source, args.channel, args.audit_windows)
        return

    channels = CHANNEL_ORDER if args.channel in ('ALL', 'BOTH') else [args.channel]
    raw_native = values.loc[a:b-1, [CHANNEL_COLUMNS[c] for c in channels]].to_numpy(dtype=float).T
    # Preserve numeric samples exactly in the source CSV; temporary interpolation
    # affects the VIEWER ONLY and any affected interval is explicitly annotated BAD.
    if np.all(~np.isfinite(raw_native), axis=1).any():
        raise ValueError('Selected span contains a channel with no finite samples')
    clean = raw_native.copy()
    preannotations = []
    for i, col in enumerate(raw_native):
        missing = ~np.isfinite(col)
        if missing.any():
            starts = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == 1)
            ends = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)) == -1)
            for s, e in zip(starts, ends):
                preannotations.append((max(0, (s - 1) / FS), min((e - s + 2) / FS, (b-a) / FS), 'BAD_missing_raw_sample', channels[i]))
            clean[i] = pd.Series(col).interpolate(limit_direction='both').to_numpy()
    # Muse native RAW sample unit is assumed microvolt for display; verify separately.
    info = mne.create_info(ch_names=channels, sfreq=FS, ch_types=['eeg']*len(channels))
    raw = mne.io.RawArray(clean * 1e-6, info, verbose=False)
    output = args.output_dir or project / 'outputs' / 'eeg' / 'manual_annotations' / f'sub{args.subject.zfill(3)}'
    output.mkdir(parents=True, exist_ok=True)
    channel_suffix = 'ALL' if args.channel == 'BOTH' else args.channel
    stem = f'{source.stem}_{channel_suffix}_span{span_idx+1}'
    fif_path = output / f'{stem}-annot.fif'
    csv_path = output / f'{stem}_intervals.csv'
    if fif_path.exists():
        annotations = mne.read_annotations(fif_path)
        print('Reopened existing annotations:', fif_path)
    else:
        annotations = mne.Annotations(onset=[], duration=[], description=[])
    if preannotations:
        for onset, duration, label, affected_channel in preannotations:
            # Prevent duplication when re-opening an already reviewed recording.
            if not fif_path.exists():
                annotations.append(onset, duration, label, ch_names=[affected_channel])
        print(f'{len(preannotations)} channel-specific missing-data spans identified (shown as BAD_missing_raw_sample)')
    raw.set_annotations(annotations)

    print(f'Showing: {args.channel}; original recording remains untouched')
    print('Press A in the MNE browser; choose BAD_clipping_suspected, REVIEW_uncertain, etc., then drag to mark an interval.')
    print('In ALL-channel view, GUI time annotations may affect all channels. For an AF7-only or AF8-only decision,')
    print('reopen using --channel AF7 or --channel AF8 and annotate there; the export will preserve channel scope.')
    print('Close the plot window to save the annotations. Use BAD_... labels for intervals to exclude.')
    if args.dry_run:
        print('Dry run successful; output folder:', output)
        return

    # Matplotlib browser needs no optional Qt dependency, works as a separate macOS window.
    mne.viz.set_browser_backend('matplotlib')
    duration = min(max(args.duration, 2.), raw.times[-1])
    if args.start is None:
        display_start = 0.0
    else:
        idx = int(np.searchsorted(segment_seconds, args.start, side='left'))
        display_start = min(idx / FS, max(0.0, raw.times[-1] - duration))
    print(f'Viewer scale = {args.scale_uv:g} assumed µV; if waveforms are still too dense, try --scale-uv 2400; for frontal detail try --channel AF7 --scale-uv 600.')
    raw.plot(start=display_start, duration=duration, n_channels=len(channels),
             scalings={'eeg':args.scale_uv*1e-6}, remove_dc=True, block=True,
             title=f'{source.name} | {args.channel} | span {span_idx+1}')

    reviewed = raw.annotations
    # The GUI defaults to global annotations. In a single-channel review, convert
    # those to explicit channel-scoped marks before saving. Existing scoped
    # annotations are preserved. ALL-mode marks remain global (all channels).
    if len(channels) == 1 and len(reviewed):
        reviewed = mne.Annotations(
            onset=reviewed.onset, duration=reviewed.duration,
            description=reviewed.description, orig_time=reviewed.orig_time,
            ch_names=[names if names else (channels[0],) for names in reviewed.ch_names],
        )
    reviewed.save(fif_path, overwrite=True)
    rows = []
    for annotation in reviewed:
        i0 = min(max(round(annotation['onset'] * FS), 0), (b-a)-1)
        i1 = min(max(round((annotation['onset']+annotation['duration']) * FS), 0), (b-a)-1)
        rows.append({
            'subject':f'sub{args.subject.zfill(3)}',
            'recording':source.name, 'channel_scope':args.channel,
            'span_number':span_idx+1, 'label':annotation['description'],
            'affected_channels': '|'.join(annotation['ch_names']) if annotation['ch_names'] else '|'.join(channels),
            'display_start_s':round(float(annotation['onset']),4),
            'display_end_s':round(float(annotation['onset']+annotation['duration']),4),
            'recording_start_s':round(float(seconds[a+i0]),4),
            'recording_end_s':round(float(seconds[a+i1]),4),
            'recording_start_timestamp':str(timestamps.iloc[a+i0]),
            'recording_end_timestamp':str(timestamps.iloc[a+i1]),
            'duration_s':round(float(annotation['duration']),4),
            'reviewer_note':'',
        })
    fields = ['subject','recording','channel_scope','span_number','label','affected_channels','display_start_s',
              'display_end_s','recording_start_s','recording_end_s',
              'recording_start_timestamp','recording_end_timestamp','duration_s','reviewer_note']
    # Preserve comments the reviewer may already have entered in the exported CSV.
    result = pd.DataFrame(rows, columns=fields)
    if csv_path.exists() and len(result):
        old = pd.read_csv(csv_path).fillna('')
        if all(k in old.columns for k in ['label', 'display_start_s', 'display_end_s', 'reviewer_note']):
            notes = {(str(r['label']), round(float(r['display_start_s']), 4),
                      round(float(r['display_end_s']), 4)): str(r['reviewer_note'])
                     for _, r in old.iterrows()}
            result['reviewer_note'] = result.apply(
                lambda r: notes.get((r['label'], round(float(r['display_start_s']),4),
                                     round(float(r['display_end_s']),4)), ''), axis=1)
    result.to_csv(csv_path, index=False)
    print(f'Saved {len(rows)} annotated intervals:')
    print('  MNE annotations:', fif_path)
    print('  Shareable review CSV:', csv_path)
    print('Scope: annotations made in ALL view affect all four channels; individual-channel runs save explicit channel scope.')
    print('Note: Existing EEG_Audit.py does NOT automatically apply these manual annotations. They must be incorporated into the final spectral analysis before calling results manually validated.')


if __name__ == '__main__':
    main()
