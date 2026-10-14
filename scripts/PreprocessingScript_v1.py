"""Python conversion of PreprocessingScript_v1.m.

EEG preprocessing, power spectral density, movement magnitude, spectrograms,
and band-power ratios for a CSV recorded at 256 Hz.

Install dependencies:
    python -m pip install numpy pandas scipy matplotlib

Run either with a file picker:
    python PreprocessingScript_v1.py

Or pass the CSV path directly:
    python PreprocessingScript_v1.py path/to/recording.csv

Add --no-show to save the outputs without opening the figure.

Outputs (next to the selected CSV):
    <recording>.tiff        2 x 4 panel analysis figure
    <recording>_Result.csv  6 x 4, whitespace-delimited numeric matrix

Important MATLAB compatibility details:
- Like MATLAB's data{2:end,idx}, the first *data row* is dropped.
- MATLAB's save(..., '-ascii') writes whitespace-separated numeric data even
  when the filename ends with .csv; that formatting is preserved here.
- MATLAB's bandstop() uses an adaptive FIR/IIR design by default. The Python
  version uses a zero-phase, 60-dB specification elliptic IIR bandstop instead.
  Consequently, filtered signals and band powers may not match exactly.
- Files shorter than four seconds use a shorter Hamming window, preserving
  NFFT=1024; this produces a single time bin for the spectrogram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, signal
from scipy.interpolate import interp1d

# ---------------- USER INPUTS ----------------
FS = 256
EEG_CHANNEL_NAMES = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
AUX_CHANNEL_NAMES = ["AUX_1", "AUX_2", "AUX_3", "AUX_4"]

WINDOW_LENGTH = 4 * FS                 # Preferred 4-second window (1024 samples)
NFFT = 1024                           # Preserve the original FFT length

# Each pair includes both endpoints, as in the original MATLAB masks.
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
    "whole": (4.0, 45.0),
}

RESULT_ROWS = (
    "theta / gamma",
    "beta / alpha",
    "theta / whole",
    "alpha / whole",
    "beta / whole",
    "gamma / whole",
)


def choose_csv_file() -> Path:
    """Use a GUI file chooser, similar to MATLAB's uigetfile()."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            filename = filedialog.askopenfilename(
                title="Select EEG CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
        finally:
            root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "The file picker could not open. Supply the CSV path instead: "
            "python PreprocessingScript_v1.py path/to/recording.csv"
        ) from exc

    if not filename:
        raise SystemExit("No CSV file selected; nothing was processed.")
    return Path(filename)


def fill_missing_linear(values: np.ndarray) -> np.ndarray:
    """Columnwise interpolation with linear end extrapolation.

    This includes MATLAB fillmissing(x, 'linear')'s default EndValues='extrap'.
    """
    filled = np.asarray(values, dtype=float).copy()
    for ch in range(filled.shape[1]):
        column = filled[:, ch]
        valid = np.flatnonzero(np.isfinite(column))
        if valid.size < 2:
            raise ValueError(
                f"Column {ch + 1} has fewer than two finite samples; "
                "linear interpolation is not possible."
            )
        if valid.size < column.size:
            interpolate = interp1d(
                valid,
                column[valid],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            missing = ~np.isfinite(column)
            column[missing] = interpolate(np.flatnonzero(missing))
    return filled


def load_eeg_and_aux(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the same CSV columns MATLAB selects with ismember()."""
    data = pd.read_csv(csv_path)

    # AF7 and AF8 are the required research channels. TP and AUX are optional.
    missing_cols = [name for name in ("RAW_AF7", "RAW_AF8") if name not in data.columns]
    if missing_cols:
        raise ValueError("CSV is missing required frontal channels: " + ", ".join(missing_cols))

    # MATLAB's find(ismember(...)) selects columns in their CSV order.
    eeg_cols = [name for name in data.columns if name in EEG_CHANNEL_NAMES]
    aux_cols = [name for name in data.columns if name in AUX_CHANNEL_NAMES]

    # MATLAB: eeg=data{2:end,idx}; aux=data{2:end,idx2};
    # This removes the first *data* row, not the header row.
    eeg = data.loc[:, eeg_cols].iloc[1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aux = data.loc[:, aux_cols].iloc[1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if eeg.shape[0] < FS:
        raise ValueError(
            f"Need at least {FS} samples (1 second at {FS} Hz) for "
            f"meaningful band-power analysis; found {eeg.shape[0]} "
            "after skipping the first data row."
        )

    return fill_missing_linear(eeg), (fill_missing_linear(aux) if aux_cols else aux), eeg_cols, aux_cols


def apply_notch_filter(eeg: np.ndarray) -> np.ndarray:
    """Close counterpart to MATLAB bandstop(eeg, [59 61], 256).

    Approximate MATLAB's default 0.85 steepness, 0.1 dB passband ripple,
    60 dB stopband attenuation with a zero-phase IIR elliptic filter.
    The internal design does not exactly duplicate MATLAB's auto FIR/IIR filter.
    """
    low, high = 59.0, 61.0
    center = (low + high) / 2
    steepness = 0.85
    fraction = 0.99 - 0.98 * steepness
    stop_low = low + fraction * (center - low)
    stop_high = high - fraction * (high - center)

    sos = signal.iirdesign(
        wp=(low, high),
        ws=(stop_low, stop_high),
        gpass=0.1,
        gstop=60.0,
        ftype="ellip",
        fs=FS,
        output="sos",
    )
    return signal.sosfiltfilt(sos, eeg, axis=0)


def compute_psd(
    values: np.ndarray, window: np.ndarray, noverlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with a 4-second window, or a shorter window if required."""
    return signal.welch(
        values,
        fs=FS,
        window=window,
        nperseg=len(window),
        noverlap=noverlap,
        nfft=NFFT,
        detrend=False,
        scaling="density",
    )


def band_power(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    """Integrate frequency rows, separately for every spectrogram time bin."""
    use = (freq >= low) & (freq <= high)
    return integrate.trapezoid(psd[use, :], x=freq[use], axis=0)


def analyze(csv_path: Path, show_figure: bool = True,
            output_dir: Path | None = None,
            save_figure: bool = True) -> tuple[Path | None, Path, np.ndarray]:
    eeg_filled, aux_filled, eeg_names, aux_names = load_eeg_and_aux(csv_path)

    # With fewer than 1024 samples, use the full available recording as one
    # window. Keep the original NFFT=1024 (zero-padding for short recordings).
    # Recordings >= 1024 samples retain the exact original 4-second settings.
    segment_length = min(WINDOW_LENGTH, len(eeg_filled))
    window = np.hamming(segment_length)  # MATLAB-compatible symmetric Hamming
    noverlap = segment_length // 2
    if segment_length < WINDOW_LENGTH:
        print(
            f"Warning: {len(eeg_filled)} samples ({len(eeg_filled) / FS:.3f} s) "
            f"after skipping the first data row; using a {segment_length}-sample "
            f"window rather than {WINDOW_LENGTH}. The spectrogram will "
            "have one time bin, so band-power ratios may be less reliable."
        )

    eeg_notch = apply_notch_filter(eeg_filled)

    # Preserve plotting calculations for compatibility with original MATLAB conversion.
    fig, axes = plt.subplots(2, 4, figsize=(20, 12), dpi=100)
    axes = axes.flat

    # 1. Original EEG power spectrum (MATLAB subplot(2,4,1)).
    for ch, channel_name in enumerate(eeg_names):
        frequencies, power = compute_psd(eeg_filled[:, ch], window, noverlap)
        with np.errstate(divide="ignore"):
            axes[0].plot(frequencies, 10 * np.log10(power), label=channel_name)
    axes[0].set_xlim(1, 120)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power Spectral Density (dB/Hz)")
    axes[0].set_title("EEG Power Spectral Density")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    # 2. EEG spectrum after 59-61 Hz notch filtering.
    for ch in range(eeg_notch.shape[1]):
        frequencies, power = compute_psd(eeg_notch[:, ch], window, noverlap)
        with np.errstate(divide="ignore"):
            axes[1].plot(frequencies, 10 * np.log10(power))
    axes[1].set_xlim(1, 120)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power Spectral Density (dB/Hz)")
    axes[1].set_title("EEG Power Spectral Density")
    axes[1].grid(True)

    # 3-4. AUX is optional: its absence must not prevent frontal power ratios.
    if aux_names:
        for ch in range(aux_filled.shape[1]):
            frequencies, power = compute_psd(aux_filled[:, ch], window, noverlap)
            with np.errstate(divide="ignore"):
                axes[2].plot(frequencies, 10 * np.log10(power), label=aux_names[ch])
        axes[2].set_xlim(1, 120)
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Power Spectral Density (dB/Hz)")
        axes[2].set_title("Movement Power Spectral Density")
        axes[2].grid(True)
        aux_detrended = signal.detrend(aux_filled, axis=0, type="linear")
        movement_mag = np.sqrt(np.sum(aux_detrended ** 2, axis=1))
        axes[3].plot(np.arange(1, movement_mag.size + 1), movement_mag)
    else:
        axes[2].text(0.5, 0.5, "AUX channels not present", ha="center", va="center")
        axes[3].text(0.5, 0.5, "Movement analysis unavailable", ha="center", va="center")

    # 5-8. Spectrograms and mean of window-by-window power ratios.
    result = np.empty((6, len(eeg_names)), dtype=float)
    for ch, channel_name in enumerate(eeg_names):
        frequencies, times, psd = signal.spectrogram(
            eeg_notch[:, ch],
            fs=FS,
            window=window,
            nperseg=segment_length,
            noverlap=noverlap,
            nfft=NFFT,
            detrend=False,
            scaling="density",
            mode="psd",
        )

        with np.errstate(divide="ignore"):
            spectrogram_db = 10 * np.log10(psd)

        ax = axes[4 + ch]
        heatmap = ax.pcolormesh(times, frequencies, spectrogram_db, shading="auto")
        ax.set_ylim(1, 100)
        ax.set_title(channel_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(heatmap, ax=ax)

        powers = {
            name: band_power(frequencies, psd, low, high)
            for name, (low, high) in BANDS.items()
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            # MATLAB takes the mean of the ratios, NOT ratio of mean powers.
            result[0, ch] = np.mean(powers["theta"] / powers["gamma"])
            result[1, ch] = np.mean(powers["beta"] / powers["alpha"])
            result[2, ch] = np.mean(powers["theta"] / powers["whole"])
            result[3, ch] = np.mean(powers["alpha"] / powers["whole"])
            result[4, ch] = np.mean(powers["beta"] / powers["whole"])
            result[5, ch] = np.mean(powers["gamma"] / powers["whole"])

    for extra_axis in range(4 + len(eeg_names), 8):
        axes[extra_axis].set_visible(False)
    fig.tight_layout(pad=2)

    # Same file naming as MATLAB, but next to the chosen CSV.
    folder = output_dir if output_dir is not None else csv_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    tiff_path = folder / (csv_path.stem + ".tiff") if save_figure else None
    results_path = folder / (csv_path.stem + "_Result.csv")
    if tiff_path is not None:
        fig.savefig(tiff_path, format="tiff", dpi=100)

    # MATLAB save(..., "-ascii") actually uses whitespace, not commas.
    np.savetxt(results_path, result, fmt="%.8e", delimiter=" ")

    if tiff_path is not None:
        print("Saved figure:", tiff_path)
    print("Saved results:", results_path)
    print("\nResults (rows as in MATLAB; columns in EEG CSV order):")
    print(pd.DataFrame(result, index=RESULT_ROWS, columns=eeg_names).to_string(float_format=lambda x: f"{x:.6g}"))

    if show_figure:
        plt.show()
    plt.close(fig)
    return tiff_path, results_path, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", help="EEG CSV file; omit to open a picker")
    parser.add_argument("--no-show", action="store_true", help="Do not open the plot window")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else choose_csv_file()
    if not csv_path.is_file():
        parser.error(f"CSV file not found: {csv_path}")
    analyze(csv_path, show_figure=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
