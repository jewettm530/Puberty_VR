#!/usr/bin/env python3
"""Manually inspect AF7 and AF8 Muse/Mind Monitor raw EEG channels with MNE.

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
    # Keep legacy sample/span coordinates when temporal columns are present so
    # existing single-channel annotations still align. Only frontal values enter MNE.
    timeline = list(values.columns) + [c for c in LEGACY_TIMELINE_COLUMNS if c in df]
    mask = df[timeline].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)
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
    if len(seconds) < 1024:
        raise ValueError('Fewer than 1024 RAW EEG samples after timestamp screening')

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

    The displayed signal is the MNE in-memory, DC-centered raw trace (not filtered).
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
                trace = trace - np.nanmedian(trace)
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
            fig.text(.10, .025, '*Amplitude assumes source units are µV; verify hardware scaling. Waveforms shown for review, not cleaned or validated.', fontsize=8)
            fig.subplots_adjust(top=.86, bottom=.09, left=.15, right=.97, hspace=.22)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='029', help='e.g. 029')
    p.add_argument('--phase', default='pre_vr', choices=['baseline','pre_vr','post_vr','recovery'])
    p.add_argument('--input', type=Path, help='Optional exact input CSV/ZIP path; overrides subject/phase lookup')
    p.add_argument('--channel', choices=['ALL', 'BOTH', 'AF7', 'AF8'], default='ALL',
                   help='ALL (default) and BOTH show AF7 and AF8; choose AF7 or AF8 to review one channel')
    p.add_argument('--start', type=float, default=None, help='Elapsed seconds since first valid RAW sample (audit coordinates)')
    p.add_argument('--duration', type=float, default=5.0, help='Visible time span in MNE browser, seconds')
    p.add_argument('--list-flags', action='store_true', help='Print representative audit flags and exit')
    p.add_argument('--audit-windows', type=Path, default=None, help='Path to EEG Audit v2 eeg_window_qc_and_power.csv')
    p.add_argument('--dry-run', action='store_true', help='Validate loading and paths without opening GUI')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--scale-uv', type=float, default=800, help='Display scaling in assumed microvolts (default 800). Try 600 for AF7/AF8-only views.')
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
    # Keep the frontal pair separate from earlier four-channel _ALL reviews.
    channel_suffix = 'AF7_AF8' if len(channels) == 2 else channels[0]
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
    for annotation, annotation_channels in zip(reviewed, reviewed.ch_names):
        i0 = min(max(round(annotation['onset'] * FS), 0), (b-a)-1)
        i1 = min(max(round((annotation['onset']+annotation['duration']) * FS), 0), (b-a)-1)
        rows.append({
            'subject':f'sub{args.subject.zfill(3)}',
            'recording':source.name, 'channel_scope':args.channel,
            'span_number':span_idx+1, 'label':annotation['description'],
            'affected_channels': '|'.join(annotation_channels) if annotation_channels else '|'.join(channels),
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
    pdf_path = output / f'{stem}_annotated_traces.pdf'
    export_annotation_pdf(raw, reviewed, pdf_path, source.name,
                          f'sub{args.subject.zfill(3)}', args.phase,
                          channels, seconds, a, fs=FS)
    print('  Annotated waveform PDF:', pdf_path)
    print(f'Saved {len(rows)} annotated intervals:')
    print('  MNE annotations:', fif_path)
    print('  Shareable review CSV:', csv_path)
    print('Scope: global annotations in the paired view cover AF7 and AF8. Individual-channel runs save explicit scope.')
    print('Note: Existing EEG_Audit.py does NOT automatically apply these manual annotations. They must be incorporated into the final spectral analysis before calling results manually validated.')


if __name__ == '__main__':
    main()
