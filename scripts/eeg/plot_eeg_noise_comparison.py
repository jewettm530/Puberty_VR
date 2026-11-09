#!/usr/bin/env python3
"""Generate paired AF7/AF8 time-series and PSD plots for local EEG recordings.

From the project root:
    python3 scripts/eeg/plot_eeg_noise_comparison.py

Plots compare the same raw and notch-filtered samples from the gap-aware
audit workflow. Each Welch PSD averages matched, non-overlapping 4-second
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


def plot_comparison(path, output, times, raw, filtered, spans, preprocessor):
    """Plot matched raw/filtered samples and return the PSD window counts."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.8), layout="constrained")
    breaks = [end for _, end in spans[:-1]]
    overview_times = np.insert(times, breaks, np.nan)
    taper = np.hamming(WINDOW_SAMPLES)
    counts = {}

    try:
        for index, (channel, (overview, detail, spectrum)) in enumerate(zip(audit.CHANNELS, axes)):
            pair = np.column_stack((raw[:, index], filtered[:, index]))
            matched = np.isfinite(pair).all(axis=1)
            starts = []
            for left, right in spans:
                edges = np.flatnonzero(np.diff(np.r_[False, matched[left:right], False]))
                for first, stop in edges.reshape(-1, 2) + left:
                    starts.extend(range(first, stop - WINDOW_SAMPLES + 1, WINDOW_SAMPLES))
            counts[channel] = len(starts)
            finite_raw = pair[np.isfinite(pair[:, 0]), 0]
            center = np.median(finite_raw) if len(finite_raw) else 0
            pair[~matched] = np.nan

            for values, (label, color, style) in zip(pair.T, STAGES):
                overview.plot(overview_times, np.insert(values - center, breaks, np.nan), style,
                              color=color, lw=0.65, label=label)
            overview.set(title=f"{channel} · Matched samples across recording",
                         xlabel="Elapsed time (s)", ylabel="Centered amplitude (raw units)",
                         xlim=(times[0], times[-1]))
            detail.set(title=f"{channel} · Matched excerpt", xlabel="Elapsed time (s)",
                       ylabel="Centered amplitude (raw units)")
            spectrum.set(title=f"{channel} · Welch PSD ({len(starts)} matched windows)",
                         xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 raw unit²/Hz)",
                         xlim=(1, 120))
            for ax in (overview, detail, spectrum):
                ax.grid(alpha=0.2)

            if not starts:
                for ax in (detail, spectrum):
                    ax.text(0.5, 0.5, "No matched 4-second window", ha="center", va="center",
                            transform=ax.transAxes)
                continue

            # Match window selection across filter stages, channel by channel.
            windows = np.stack([
                pair[start:start + WINDOW_SAMPLES].T for start in starts
            ])
            frequencies, density = preprocessor.compute_psd(windows, taper, 0)
            with np.errstate(divide="ignore"):
                psd_db = 10 * np.log10(density.mean(axis=0))

            excerpt_start = starts[len(starts) // 2] + int(audit.FS)
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
            times, raw, _, _ = audit.load(path)
            duration = times[-1] - times[0]
            dt = np.diff(times)
            positive = dt[dt > 0]
            median_hz = 1 / np.median(positive) if len(positive) else np.nan
            average_hz = (len(times) - 1) / duration if duration > 0 else 0
            row.update(median_hz=median_hz, average_hz=average_hz)
            if len(times) < WINDOW_SAMPLES or not (230 <= median_hz <= 290 and average_hz >= 180):
                row.update(status="SKIPPED_RATE_OR_LENGTH",
                           reason="Needs at least 4 seconds of full-rate EEG (audit rate screen).")
            else:
                spans, _ = audit.identify_spans(times)
                filtered = audit.prefilter_valid_spans(raw, spans)
                counts = plot_comparison(path, args.output_dir / plot, times, raw,
                                         filtered, spans, audit.PREPROCESSOR)
                row.update(af7_psd_windows=counts["AF7"], af8_psd_windows=counts["AF8"])
                if any(counts.values()):
                    status = "PLOTTED" if all(counts.values()) else "PLOTTED_PARTIAL"
                    row.update(status=status, plot=plot.as_posix())
                    if status == "PLOTTED_PARTIAL":
                        row["reason"] = "One channel has no matched 4-second window."
                else:
                    row.update(status="SKIPPED_NO_MATCHED_WINDOWS",
                               reason="Neither channel has a matched 4-second window.")
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
