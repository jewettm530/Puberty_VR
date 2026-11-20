"""Shared AF7/AF8 preprocessing and fixed two-second Welch analysis.

Run from the project root:
    python3 scripts/eeg/preprocessing.py recording.csv --no-show

Subjects 039-052 use only the second half of each recording, by elapsed time.
Other subjects use the complete recording. A full 512-sample epoch at 256 Hz
is required. Epochs advance by 256 samples (50% overlap). Every PSD uses the
same fixed Hann taper, original 1024-point FFT and density scaling in microvolt^2/Hz.
Finite contiguous spans are not joined across gaps or the eyes-open boundary.

Muse RAW values are interpreted as microvolts with a 725-microvolt DC offset.
Set --raw-offset-uv 0 for already signed microvolt recordings. An epoch is
rejected if either its offset-corrected input or its notch-filtered signal
exceeds +/-200 microvolts. Rejected epochs remain visible in the saved
preprocessed dataset with QC flags, preserving time for later manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, signal

FS = 256
WINDOW_SEC = 2.0
STEP_SEC = 1.0
WINDOW_LENGTH = 512
NFFT = 1024
CHANNELS = ("AF7", "AF8")
EEG_CHANNEL_NAMES = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
AUX_CHANNEL_NAMES = ["AUX_1", "AUX_2", "AUX_3", "AUX_4"]
VERSION = "4.2-eyes-open-welch2s-original-layout"
TAPER = signal.windows.hann(WINDOW_LENGTH, sym=False)
BANDS = {"theta": (4., 8.), "alpha": (8., 13.), "beta": (13., 30.),
         "gamma": (30., 45.), "whole": (4., 45.)}
RESULT_ROWS = ("theta / gamma", "beta / alpha", "theta / whole",
               "alpha / whole", "beta / whole", "gamma / whole")
QC_COLUMNS = [
    "channel", "start_s", "end_s", "window_index", "stop_index", "timing_ok",
    "theta", "alpha", "beta", "gamma", "whole", "high_beta", "hf_warning",
    "slow_1_4_warning", "ratio", "hf_beta_indicator", "raw_ptp", "raw_max_jump",
    "boundary_fraction_observed", "filtered_ptp", "max_abs_input_uv",
    "max_abs_filtered_uv", "amplitude_rejected", "manual_rejected",
    "hard_reasons", "warning_reasons", "automatic_pass", "hard_pass",
    "strict_pass", "paired_pass",
]


class IneligibleRecording(ValueError):
    """A recording cannot supply fixed full-rate epochs."""


def read_csv(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")
                     and not n.startswith("__MACOSX/") and not Path(n).name.startswith("._")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one EEG CSV inside {path.name}")
            with archive.open(names[0]) as handle:
                return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def recording_identity(path):
    path = Path(path)
    match = re.match(r"^(\d+)_(.+)_eeg$", path.stem, re.I)
    if match:
        return match[1].zfill(3), match[2].lower()
    folder = next((re.fullmatch(r"sub_?(\d+)", part, re.I)
                   for part in reversed(path.parts) if re.fullmatch(r"sub_?\d+", part, re.I)), None)
    if folder is None:
        raise ValueError("Use a subject-prefixed recording name such as 039_pre_vr_eeg.csv")
    return folder[1].zfill(3), re.sub(r"_eeg$", "", path.stem, flags=re.I).lower()


def load_recording(path):
    """Read CSV/ZIP once and retain strictly increasing source times and gaps."""
    frame = read_csv(path)
    missing = {"TimeStamp", "RAW_AF7", "RAW_AF8"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing EEG columns: {sorted(missing)}")
    columns = [c for c in ("RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10") if c in frame]
    has_sample = frame[columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    stamps = pd.to_datetime(frame["TimeStamp"], errors="coerce")
    usable = np.flatnonzero((has_sample & stamps.notna()).to_numpy())
    retained = []
    last_time = None
    for index in usable:
        when = stamps.iloc[index]
        if last_time is None or when > last_time:
            retained.append(index)
            last_time = when
    if len(retained) < 2:
        raise IneligibleRecording("Insufficient timestamped raw samples")
    times = stamps.iloc[retained].reset_index(drop=True)
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    eeg_names = [c for c in frame if c in EEG_CHANNEL_NAMES]
    aux_names = [c for c in frame if c in AUX_CHANNEL_NAMES]
    eeg_raw = frame.iloc[retained][eeg_names].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aux = frame.iloc[retained][aux_names].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw = frame.iloc[retained][["RAW_AF7", "RAW_AF8"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return {"times": seconds, "timestamps": times.astype(str).to_numpy(dtype=str),
            "raw": raw, "eeg_raw": eeg_raw, "eeg_names": eeg_names,
            "aux": aux, "aux_names": aux_names, "source_rows": np.asarray(retained),
            "raw_export_rows": len(frame),
            "out_of_order_or_duplicate_rows": len(usable) - len(retained)}


def identify_spans(times, gap_factor=3.5):
    cuts = np.flatnonzero(np.diff(times) > gap_factor / FS) + 1
    return [(int(a), int(b)) for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(times)])]


def apply_notch_filter(values):
    low, high = 59., 61.
    fraction = .99 - .98 * .85
    sos = signal.iirdesign(
        wp=(low, high), ws=(low + fraction, high - fraction),
        gpass=.1, gstop=60., ftype="ellip", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def prefilter_valid_spans(raw, spans):
    """Filter finite runs after cropping, with no interpolation across missing data."""
    filtered = np.full(raw.shape, np.nan)
    for a, b in spans:
        for channel in range(raw.shape[1]):
            finite = np.isfinite(raw[a:b, channel])
            edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(int)))
            for left, right in edges.reshape(-1, 2):
                if right - left >= WINDOW_LENGTH:
                    filtered[a+left:a+right, channel] = apply_notch_filter(
                        raw[a+left:a+right, channel, None])[:, 0]
    return filtered


def compute_psd(values, window=None, noverlap=None):
    """Use fixed 512-sample periodic Hann windows and the original 1024 FFT and no detrending.

    Optional arguments accept only the fixed taper and configured overlap.
    They support existing helper callers without allowing variable-length epochs.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[-1] < WINDOW_LENGTH:
        raise IneligibleRecording("Welch PSD requires at least 512 samples (2 seconds)")
    if window is not None and not np.array_equal(np.asarray(window), TAPER):
        raise ValueError("Welch taper must be the fixed 512-sample Hann window")
    overlap = WINDOW_LENGTH // 2 if noverlap is None else noverlap
    if overlap not in (0, WINDOW_LENGTH // 2):
        raise ValueError("Supported overlaps are 0 or 256 samples")
    return signal.welch(values, fs=FS, window=TAPER, nperseg=WINDOW_LENGTH,
                        noverlap=overlap, nfft=NFFT, detrend=False,
                        scaling="density", average="mean", axis=-1)


def band_power(frequencies, psd, low, high):
    use = (frequencies >= low) & (frequencies <= high)
    return integrate.trapezoid(np.asarray(psd)[use], x=frequencies[use], axis=0)


def read_manual_intervals(directory, metadata):
    """Only post-preprocessing CSVs with the exact processing identity are applied."""
    intervals = []
    ignored = []
    if directory is None or not Path(directory).is_dir():
        return intervals, ignored
    for path in sorted(Path(directory).glob("*_intervals.csv")):
        if not path.name.startswith(metadata["recording_stem"] + "_"):
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        required = {"preprocessing_id", "label", "affected_channels",
                    "recording_start_s", "recording_end_s"}
        if not required.issubset(frame.columns):
            ignored.append(path.name)
            continue
        if not frame.empty and not frame.preprocessing_id.eq(metadata["preprocessing_id"]).all():
            ignored.append(path.name)
            continue
        for row in frame.itertuples(index=False):
            if not str(row.label).startswith("BAD_") or str(row.label).startswith("BAD_auto_"):
                continue
            start, end = float(row.recording_start_s), float(row.recording_end_s)
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError(f"Invalid manual annotation interval in {path.name}")
            channels = str(row.affected_channels).split("|")
            if not set(channels).issubset(CHANNELS):
                raise ValueError(f"Unknown annotation channel scope in {path.name}")
            intervals.append((start, end, channels))
    return intervals, ignored


def process_recording(path, *, raw_offset_uv=725., amplitude_limit_uv=200.,
                      min_average_rate=180., gap_factor=3.5, timing_tolerance=.08,
                      flat_std=.05, flat_run_ms=125., bound_fraction=.01,
                      manual_dir=None):
    """The sole implementation of cropping, filtering, epoch QC and PSD."""
    path = Path(path)
    if not np.isfinite([raw_offset_uv, amplitude_limit_uv, min_average_rate,
                        gap_factor, timing_tolerance, flat_std, flat_run_ms,
                        bound_fraction]).all():
        raise ValueError("Processing settings must be finite")
    if min(amplitude_limit_uv, gap_factor, flat_run_ms) <= 0 or timing_tolerance < 0:
        raise ValueError("Invalid processing threshold")
    subject, phase = recording_identity(path)
    loaded = load_recording(path)
    t = loaded["times"]
    median_hz = float(1 / np.median(np.diff(t)))
    average_hz = float((len(t)-1) / (t[-1]-t[0]))
    if not 230 <= median_hz <= 290 or average_hz < min_average_rate:
        raise IneligibleRecording(
            f"Not full-rate: median {median_hz:.2f}, average {average_hz:.2f} Hz")
    # Half of acquisition duration, not half of CSV rows (metadata rows may be mixed in).
    cutoff = float((t[-1] + 1 / FS) / 2) if 39 <= int(subject) <= 52 else 0.
    first = int(np.searchsorted(t, cutoff, side="left"))
    eeg_names = loaded["eeg_names"]
    diagnostic_channels = [name.removeprefix("RAW_") for name in eeg_names]
    raw = loaded["eeg_raw"][first:]
    frontal_indices = [diagnostic_channels.index(channel) for channel in CHANNELS]
    times = t[first:]
    if len(times) < WINDOW_LENGTH:
        raise IneligibleRecording("Selected eyes-open interval has fewer than 512 samples")
    spans = identify_spans(times, gap_factor)
    centered = raw - raw_offset_uv
    filtered = prefilter_valid_spans(centered, spans)
    metadata = {
        "version": VERSION, "subject_id": subject, "phase": phase,
        "recording": path.name, "recording_stem": path.stem,
        "fs_hz": FS, "window_s": WINDOW_SEC, "step_s": STEP_SEC, "nfft": NFFT,
        "taper": "fixed periodic Hann", "detrend": False,
        "diagnostic_channels": eeg_names, "aux_channels": loaded["aux_names"],
        "psd_units": "uV^2/Hz",
        "raw_offset_uv": raw_offset_uv, "amplitude_limit_uv": amplitude_limit_uv,
        "gap_factor": gap_factor, "timing_tolerance": timing_tolerance,
        "flat_std": flat_std, "flat_run_ms": flat_run_ms, "bound_fraction": bound_fraction,
        "selection": "second_half_eyes_open" if cutoff else "whole_recording",
        "selection_cutoff_s": cutoff, "analysis_start_s": float(times[0]),
        "analysis_end_s": float(times[-1] + 1 / FS),
        "original_duration_s": float(t[-1] + 1 / FS),
        "estimated_median_hz": median_hz, "observed_average_hz": average_hz,
        "raw_export_rows": loaded["raw_export_rows"],
        "raw_sample_rows": len(t), "selected_sample_rows": len(times),
        "out_of_order_or_duplicate_rows": loaded["out_of_order_or_duplicate_rows"],
        "timestamp_gaps": len(spans)-1,
    }
    # Hash source signals and source timing as well as settings. Independent of manual edits.
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    identity.update(loaded["eeg_raw"].tobytes())
    identity.update(t.tobytes())
    identity.update(loaded["timestamps"].tobytes())
    metadata["preprocessing_id"] = identity.hexdigest()
    manual, ignored = read_manual_intervals(manual_dir, metadata)
    metadata["ignored_annotation_files"] = ignored
    metadata["manual_bad_intervals"] = len(manual)
    metadata["analysis_id"] = hashlib.sha256(
        (metadata["preprocessing_id"]+json.dumps(manual, sort_keys=True)).encode()).hexdigest()
    rows, spectra = [], []
    frequencies = np.fft.rfftfreq(NFFT, 1/FS)
    bands = {**BANDS, "high_beta": (20., 30.), "hf_warning": (35., 45.),
             "slow_1_4_warning": (1., 4.)}
    for a, b in spans:
        for start in range(a, b-WINDOW_LENGTH+1, int(STEP_SEC*FS)):
            end = start + WINDOW_LENGTH
            duration = times[end-1] - times[start]
            timing_ok = abs(duration - (WINDOW_LENGTH-1)/FS) <= timing_tolerance
            window_spectra = np.full((len(diagnostic_channels), len(frequencies)), np.nan)
            for ci, channel in enumerate(diagnostic_channels):
                before, clean = centered[start:end, ci], filtered[start:end, ci]
                source = raw[start:end, ci]
                reasons = []
                if not timing_ok:
                    reasons.append("irregular_timestamps")
                if not np.isfinite(before).all():
                    reasons.append("missing_samples")
                if not np.isfinite(clean).all():
                    reasons.append("filter_unavailable")
                if np.any(np.abs(before) > amplitude_limit_uv) or np.any(
                        np.abs(clean) > amplitude_limit_uv):
                    reasons.append("amplitude_exceeds_limit_uv")
                if np.isfinite(before).all():
                    if np.std(before) < flat_std:
                        reasons.append("flatline")
                    repeated = np.abs(np.diff(before)) < .01
                    edges = np.flatnonzero(np.diff(np.r_[False, repeated, False].astype(int)))
                    if len(edges) and max(np.diff(edges.reshape(-1, 2), axis=1)[:, 0]) >= flat_run_ms/1000*FS:
                        reasons.append("constant_run")
                boundary = np.mean(np.isclose(source, 0, atol=.01) |
                                   np.isclose(source, 1450, atol=.01))
                if raw_offset_uv == 725. and boundary > bound_fraction:
                    reasons.append("boundary_values")
                automatic_pass = not reasons
                manual_rejected = any(
                    times[start] < hi and times[end-1]+1/FS > lo and channel in scope
                    for lo, hi, scope in manual)
                if manual_rejected:
                    reasons.append("manual_bad_interval")
                row = {key: np.nan for key in QC_COLUMNS}
                row.update(
                    channel=channel, start_s=float(times[start]),
                    end_s=float(times[end-1]+1/FS), window_index=start, stop_index=end,
                    timing_ok=timing_ok, automatic_pass=automatic_pass,
                    amplitude_rejected="amplitude_exceeds_limit_uv" in reasons,
                    manual_rejected=manual_rejected, hard_reasons="|".join(reasons),
                    warning_reasons="", hard_pass=not reasons, strict_pass=not reasons,
                    raw_ptp=float(np.ptp(before)) if np.isfinite(before).all() else np.nan,
                    raw_max_jump=float(np.max(np.abs(np.diff(before)))) if np.isfinite(before).all() else np.nan,
                    boundary_fraction_observed=boundary,
                    filtered_ptp=float(np.ptp(clean)) if np.isfinite(clean).all() else np.nan,
                    max_abs_input_uv=float(np.nanmax(np.abs(before))) if np.isfinite(before).any() else np.nan,
                    max_abs_filtered_uv=float(np.nanmax(np.abs(clean))) if np.isfinite(clean).any() else np.nan)
                if not reasons:
                    _, psd = compute_psd(clean + raw_offset_uv, noverlap=0)
                    window_spectra[ci] = psd
                    row.update({name: float(band_power(frequencies, psd[:, None], lo, hi)[0])
                                for name, (lo, hi) in bands.items()})
                    row["ratio"] = row["beta"]/row["alpha"] if row["alpha"] > 0 else np.nan
                    row["hf_beta_indicator"] = row["hf_warning"]/row["beta"] if row["beta"] > 0 else np.nan
                rows.append(row)
            spectra.append(window_spectra)
    if not rows:
        raise IneligibleRecording("No complete two-second window within a contiguous span")
    diagnostic_qc = pd.DataFrame(rows, columns=QC_COLUMNS)
    qc = diagnostic_qc.loc[diagnostic_qc.channel.isin(CHANNELS)].copy()
    paired = qc.groupby("window_index").hard_pass.all()
    qc["paired_pass"] = qc.hard_pass & qc.window_index.map(paired)
    spectra = np.asarray(spectra)
    return {"metadata": metadata, "times": times, "raw_uv": centered[:, frontal_indices],
            "filtered_uv": filtered[:, frontal_indices], "timestamps": loaded["timestamps"][first:],
            "source_rows": loaded["source_rows"][first:], "spans": np.asarray(spans),
            "windows": qc, "frequencies": frequencies, "psd": spectra[:, frontal_indices],
            "diagnostic_raw": raw, "diagnostic_filtered": filtered + raw_offset_uv,
            "diagnostic_windows": diagnostic_qc, "diagnostic_psd": spectra,
            "eeg_names": eeg_names, "aux": loaded["aux"][first:],
            "aux_names": loaded["aux_names"]}


def save_preprocessed(result, path):
    """Preserve samples/time and all QC decisions for post-preprocessing review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(result["metadata"]),
                            times=result["times"], raw_uv=result["raw_uv"],
                            filtered_uv=result["filtered_uv"], timestamps=result["timestamps"],
                            source_rows=result["source_rows"], spans=result["spans"])
    temporary.replace(path)
    result["windows"].to_csv(path.with_name(path.stem + "_WindowQC.csv"), index=False)


def load_preprocessed(path):
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key].copy() for key in stored.files if key != "metadata"}
        result["metadata"] = json.loads(str(stored["metadata"]))
    if result["metadata"]["version"] != VERSION:
        raise ValueError("Rebuild EEG with the current preprocessing.py before manual review")
    result["windows"] = pd.read_csv(Path(path).with_name(Path(path).stem + "_WindowQC.csv"),
                                   keep_default_na=False)
    return result


def diagnostic_psds(result):
    """Use each channel's accepted two-second windows for both PSD curves."""
    before, after = [], []
    qc = result["diagnostic_windows"]
    for ci, name in enumerate(result["eeg_names"]):
        rows = qc[(qc.channel == name.removeprefix("RAW_")) & qc.hard_pass]
        raw_psd = np.full(len(result["frequencies"]), np.nan)
        filtered_psd = raw_psd.copy()
        if len(rows):
            epochs = np.stack([result["diagnostic_raw"][int(row.window_index):int(row.stop_index), ci]
                               for row in rows.itertuples()])
            _, values = compute_psd(epochs, noverlap=0)
            raw_psd = values.mean(axis=0)
            available = result["diagnostic_psd"][:, ci]
            filtered_psd = available[np.isfinite(available).all(axis=1)].mean(axis=0)
        before.append(raw_psd)
        after.append(filtered_psd)
    return before, after


def plot_spectrogram_axes(axes, result):
    """Keep the original per-channel heatmap style and automatic color range."""
    qc = result["diagnostic_windows"]
    frequencies = result["frequencies"]
    starts = qc.loc[qc.channel == result["eeg_names"][0].removeprefix("RAW_"),"start_s"].to_numpy()
    half_bin = (frequencies[1]-frequencies[0])/2
    for ci, (ax, name) in enumerate(zip(axes, result["eeg_names"])):
        with np.errstate(divide="ignore"):
            density_db = np.ma.masked_invalid(10*np.log10(result["diagnostic_psd"][:,ci].T))
        finite = density_db.compressed()
        vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        splits = np.flatnonzero(np.diff(starts) > STEP_SEC+.1)+1
        for left,right in zip(np.r_[0,splits],np.r_[splits,len(starts)]):
            times = np.r_[starts[left:right],starts[right-1]+STEP_SEC]
            heatmap = ax.pcolormesh(times,np.r_[frequencies-half_bin,frequencies[-1]+half_bin],
                                   density_db[:,left:right],shading="flat",cmap="viridis",
                                   vmin=vmin,vmax=vmax,rasterized=True)
        ax.set(title=name,xlabel="Time (s)",ylabel="Frequency (Hz)",ylim=(1,100),
               xlim=(starts[0],starts[-1]+STEP_SEC))
        ax.figure.colorbar(heatmap,ax=ax,label="PSD (dB/Hz)")
    for ax in axes[len(result["eeg_names"]):]:
        ax.set_visible(False)


def plot_diagnostics(result):
    """Restore the original 2-by-4 EEG, movement and spectrogram layout."""
    fig, grid = plt.subplots(2,4,figsize=(20,12),dpi=100)
    axes = grid.ravel()
    before, after = diagnostic_psds(result)
    frequencies = result["frequencies"]
    for ci, name in enumerate(result["eeg_names"]):
        with np.errstate(divide="ignore"):
            axes[0].plot(frequencies,10*np.log10(before[ci]),label=name)
            axes[1].plot(frequencies,10*np.log10(after[ci]))
    for ax,title in zip(axes[:2],("EEG Power Spectral Density","Notch-filtered EEG PSD")):
        ax.set(xlim=(1,120),xlabel="Frequency (Hz)",
               ylabel="Power Spectral Density (dB/Hz)",title=title)
        ax.grid(True)
    axes[0].legend(loc="best")
    if result["aux_names"]:
        aux = result["aux"].copy()
        # Keep the original linear fill for movement diagnostics only.
        from scipy.interpolate import interp1d
        for ci in range(aux.shape[1]):
            valid = np.flatnonzero(np.isfinite(aux[:,ci]))
            if len(valid) >= 2:
                fill = interp1d(valid,aux[valid,ci],bounds_error=False,fill_value="extrapolate")
                missing = ~np.isfinite(aux[:,ci])
                aux[missing,ci] = fill(np.flatnonzero(missing))
        for ci, name in enumerate(result["aux_names"]):
            spectra = []
            for left,right in result["spans"]:
                for start in range(left,right-WINDOW_LENGTH+1,int(STEP_SEC*FS)):
                    values = aux[start:start+WINDOW_LENGTH,ci]
                    if np.isfinite(values).all():
                        _, density = compute_psd(values,noverlap=0)
                        spectra.append(density)
            if spectra:
                with np.errstate(divide="ignore"):
                    axes[2].plot(frequencies,10*np.log10(np.mean(spectra,axis=0)),label=name)
        axes[2].set(xlim=(1,120),xlabel="Frequency (Hz)",
                    ylabel="Power Spectral Density (dB/Hz)",title="Movement PSD")
        axes[2].grid(True)
        if np.isfinite(aux).all():
            magnitude = np.sqrt(np.sum(signal.detrend(aux,axis=0,type="linear")**2,axis=1))
            axes[3].plot(np.arange(1,len(magnitude)+1),magnitude)
        else:
            axes[3].text(.5,.5,"Movement data unavailable",ha="center",va="center")
    else:
        axes[2].text(.5,.5,"AUX channels not present",ha="center",va="center")
        axes[3].text(.5,.5,"Movement analysis unavailable",ha="center",va="center")
    plot_spectrogram_axes(axes[4:],result)
    fig.tight_layout(pad=2)
    return fig


def export_outputs(result, csv_path, output_dir, *, save_figure=False,
                   save_spectrogram=True, show_figure=False):
    """Export accepted-epoch PSDs and legacy-compatible band tables."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    meta, qc, density = result["metadata"], result["windows"], result["psd"]
    frequencies = result["frequencies"]
    frontal, psd_rows = [], []
    matrix = np.full((6, len(result["eeg_names"])), np.nan)
    pairs = (("theta", "gamma"), ("beta", "alpha"), ("theta", "whole"),
             ("alpha", "whole"), ("beta", "whole"), ("gamma", "whole"))
    for ci, channel in enumerate(CHANNELS):
        accepted = qc[(qc.channel == channel) & qc.hard_pass]
        mean_psd = np.full(len(frequencies), np.nan)
        if len(accepted):
            mean_psd = density[:, ci][np.isfinite(density[:, ci]).all(axis=1)].mean(axis=0)
        psd_rows.extend({"channel": channel, "frequency_hz": float(f),
                         "psd_uv2_per_hz": float(p), "n_windows": len(accepted)}
                        for f, p in zip(frequencies, mean_psd))
        for band in ("theta", "alpha", "beta"):
            power = float(accepted[band].mean()) if len(accepted) else np.nan
            frontal.append({"channel": channel, "band": band, "power": power,
                            "log10_power": float(np.log10(power)) if power > 0 else np.nan,
                            "n_windows": len(accepted), "pipeline_version": VERSION,
                            "preprocessing_id": meta["preprocessing_id"],
                            "analysis_id": meta["analysis_id"],
                            "selection": meta["selection"], "analysis_start_s": meta["analysis_start_s"],
                            "analysis_end_s": meta["analysis_end_s"],
                            "window_s": WINDOW_SEC, "psd_units": "uV^2/Hz"})
    for ci, name in enumerate(result["eeg_names"]):
        rows = result["diagnostic_windows"]
        accepted = rows[(rows.channel == name.removeprefix("RAW_")) & rows.hard_pass]
        for row, (num, den) in enumerate(pairs):
            values = accepted[num]/accepted[den]
            matrix[row,ci] = float(values.replace([np.inf,-np.inf],np.nan).mean())
    pd.DataFrame(frontal).to_csv(folder / (stem+"_FrontalLogPower.csv"), index=False)
    pd.DataFrame(psd_rows).to_csv(folder / (stem+"_PSD.csv"), index=False)
    result_path = folder / (stem+"_Result.csv")
    np.savetxt(result_path, matrix, fmt="%.8e")
    qc.to_csv(folder / (stem+"_WindowQC.csv"), index=False)
    (folder / (stem+"_Processing.json")).write_text(json.dumps(meta, indent=2)+"\n")
    tiff_path = None
    if save_spectrogram or save_figure or show_figure:
        diagnostic = plot_diagnostics(result)
        if save_spectrogram or save_figure:
            diagnostic.savefig(folder/(stem+"_PSD.png"),dpi=100)
        if save_figure:
            tiff_path = folder/(stem+".tiff")
            diagnostic.savefig(tiff_path,format="tiff",dpi=100)
        if save_spectrogram or save_figure:
            n_channels = len(result["eeg_names"])
            fig, axes = plt.subplots(1,n_channels,figsize=(5*n_channels,4.5),
                                     squeeze=False,layout="constrained")
            plot_spectrogram_axes(axes[0],result)
            fig.savefig(folder/(stem+"_Spectrogram.png"),dpi=150)
            plt.close(fig)
        if show_figure:
            plt.show()
        plt.close(diagnostic)
    return tiff_path, result_path, matrix


def analyze(csv_path, show_figure=True, output_dir=None, save_figure=True, *,
            save_spectrogram=False, processed_path=None, **settings):
    result = process_recording(csv_path, **settings)
    folder = Path(output_dir) if output_dir else Path(csv_path).parent
    if processed_path is None:
        processed_path = folder / (Path(csv_path).stem+"_Preprocessed.npz")
    save_preprocessed(result, processed_path)
    return export_outputs(result, csv_path, folder, save_figure=save_figure,
                          save_spectrogram=save_spectrogram, show_figure=show_figure)


def choose_csv_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    name = filedialog.askopenfilename(filetypes=[("EEG", "*.csv *.zip")])
    root.destroy()
    if not name:
        raise SystemExit("No EEG file selected")
    return Path(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--raw-offset-uv", type=float, default=725.)
    parser.add_argument("--amplitude-limit-uv", type=float, default=200.)
    args = parser.parse_args()
    path = args.csv if args.csv else choose_csv_file()
    analyze(path, show_figure=not args.no_show, save_spectrogram=True,
            raw_offset_uv=args.raw_offset_uv, amplitude_limit_uv=args.amplitude_limit_uv)


if __name__ == "__main__":
    main()
