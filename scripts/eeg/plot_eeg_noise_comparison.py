#!/usr/bin/env python3
"""Generate paired AF7/AF8 time-series and PSD plots for local EEG recordings.

From the project root:
    python3 scripts/eeg/plot_eeg_noise_comparison.py

Plots compare the same raw and notch-filtered samples from the gap-aware
audit workflow. Each Welch PSD averages matched, accepted 2-second
windows. The command writes only figures and a coverage CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

import EEG_Audit as audit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_SAMPLES = int(audit.FS * audit.WINDOW_SEC)
STAGES = (("Before notch", "#59616e", "-"), ("After notch", "#007a78", "--"))


def plot_comparison(path, output, times, raw, filtered, spans, preprocessor, qc):
    """Plot matched raw/filtered samples and return the PSD window counts."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8), layout="constrained")
    breaks = [end for _, end in spans[:-1]]
    overview_times = np.insert(times, breaks, np.nan)
    taper = preprocessor.TAPER
    counts = {}

    try:
        for index, (channel, (overview, detail, spectrum)) in enumerate(zip(audit.CHANNELS, axes)):
            pair = np.column_stack((raw[:, index], filtered[:, index]))
            matched = np.isfinite(pair).all(axis=1)
            starts = qc.loc[(qc.channel == channel) & qc.hard_pass,
                            'window_index'].astype(int).tolist()
            counts[channel] = len(starts)
            finite_raw = pair[np.isfinite(pair[:, 0]), 0]
            center = np.median(finite_raw) if len(finite_raw) else 0
            pair[~matched] = np.nan

            for values, (label, color, style) in zip(pair.T, STAGES):
                overview.plot(overview_times, np.insert(values - center, breaks, np.nan), style,
                              color=color, lw=0.65, label=label)
            overview.set(title=f"{channel} · Matched samples across recording",
                         xlabel="Elapsed time (s)", ylabel="Amplitude (µV)",
                         xlim=(times[0], times[-1]))
            detail.set(title=f"{channel} · Matched excerpt", xlabel="Elapsed time (s)",
                       ylabel="Amplitude (µV)")
            spectrum.set(title=f"{channel} · Welch PSD ({len(starts)} matched windows)",
                         xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 µV²/Hz)",
                         xlim=(1, 120))
            for ax in (overview, detail, spectrum):
                ax.grid(alpha=0.2)

            if not starts:
                for ax in (detail, spectrum):
                    ax.text(0.5, 0.5, "No matched 2-second window", ha="center", va="center",
                            transform=ax.transAxes)
                continue

            # Match window selection across filter stages, channel by channel.
            windows = np.stack([
                pair[start:start + WINDOW_SAMPLES].T for start in starts
            ])
            frequencies, density = preprocessor.compute_psd(windows, taper, 0)
            with np.errstate(divide="ignore"):
                psd_db = 10 * np.log10(density.mean(axis=0))

            excerpt_start = starts[len(starts) // 2]
            excerpt_stop = excerpt_start + 2 * int(audit.FS)
            excerpt = slice(excerpt_start, excerpt_stop)
            overview.axvspan(times[excerpt_start], times[excerpt_stop - 1], color="#ccd5df", alpha=0.55)
            for values, db, (label, color, style) in zip(pair.T, psd_db, STAGES):
                detail.plot(times[excerpt], values[excerpt] - center, style,
                            color=color, lw=0.85, label=label)
                spectrum.plot(frequencies, np.where(np.isfinite(db), db, np.nan),
                              style, color=color, lw=1.0, label=label)
            detail.xaxis.set_major_locator(MaxNLocator(nbins=5))
            spectrum.axvspan(59, 61, color="#d39a37", alpha=0.25, label="59–61 Hz")
            spectrum.legend(loc="best", fontsize=8)

        if not any(counts.values()):
            return counts
        fig.suptitle(f"{path.stem}  |  60 Hz notch comparison", fontsize=15, fontweight="bold")
        fig.supxlabel("Time axes use recording timestamps. PSD uses nominal 256 Hz. "
                     "Shading on the full traces locates the excerpts.", fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
        return counts
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / "subject_data")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "eeg" / "noise_comparison")
    parser.add_argument("--raw-offset-uv", type=float, default=725.)
    parser.add_argument("--amplitude-limit-uv", type=float, default=200.)
    args = parser.parse_args()
    if not args.subjects_dir.is_dir():
        parser.error(f"Raw subject directory not found: {args.subjects_dir}")

    paths = {}
    for path in audit.discover(args.subjects_dir):
        # Prefer the CSV if the same recording is stored in both formats.
        key = (path.parent, path.stem.lower())
        if key not in paths or path.suffix.lower() == ".csv":
            paths[key] = path
    if not paths:
        parser.error(f"No EEG CSV or ZIP recordings found in {args.subjects_dir}")

    # The audit's gap-aware filtering calls the project's shared notch function.
    audit.PREPROCESSOR = audit.load_preprocessing_module(PROJECT_ROOT)
    report = []
    for path in paths.values():
        relative = path.relative_to(args.subjects_dir)
        plot = relative.with_name(path.stem + "_NoiseComparison.png")
        row = {"recording": relative.as_posix(), "median_hz": np.nan, "average_hz": np.nan,
               "af7_psd_windows": 0, "af8_psd_windows": 0, "status": "", "plot": "", "reason": ""}
        try:
            data = audit.PREPROCESSOR.process_recording(
                path, raw_offset_uv=args.raw_offset_uv, amplitude_limit_uv=args.amplitude_limit_uv,
                manual_dir=PROJECT_ROOT/'outputs'/'eeg'/'manual_annotations'/(
                    'sub'+audit.PREPROCESSOR.recording_identity(path)[0]))
            times, raw, filtered = data['times'], data['raw_uv'], data['filtered_uv']
            spans = data['spans']
            qc = data['windows']
            meta = data['metadata']
            row.update(median_hz=meta['estimated_median_hz'],average_hz=meta['observed_average_hz'])
            counts = plot_comparison(path,args.output_dir/plot,times,raw,filtered,spans,audit.PREPROCESSOR,qc)
            row.update(af7_psd_windows=counts['AF7'],af8_psd_windows=counts['AF8'])
            row.update(status='PLOTTED' if all(counts.values()) else 'PLOTTED_PARTIAL' if any(counts.values())
                       else 'SKIPPED_NO_MATCHED_WINDOWS',
                       plot=plot.as_posix() if any(counts.values()) else '')
        except (OSError, ValueError, FloatingPointError, BadZipFile) as exc:
            row.update(status="ERROR", reason=f"{type(exc).__name__}: {exc}")
        print(f"{relative}: {row['status']}" + (f" ({row['reason']})" if row["reason"] else ""),
              flush=True)
        report.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = args.output_dir / "recording_coverage.csv"
    pd.DataFrame(report).to_csv(coverage, index=False)
    print(f"Coverage report: {coverage}")
    return 0 if any(row["plot"] for row in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
