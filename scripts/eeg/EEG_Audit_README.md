# EEG Audit — Puberty & VR

Save `EEG_Audit.py` to `Puberty_VR/scripts/` beside `PreprocessingScript_v1.py` (which remains unchanged).

```bash
cd /Users/maddiemac/Puberty_VR
python3 -m pip install numpy pandas scipy matplotlib
python3 scripts/EEG_Audit.py
```

By default it finds CSV and ZIP EEG exports under `data/raw/subject_data/sub###/`, and writes results to `outputs/eeg/audit/`. It never modifies inputs. Example of one subject:

```bash
python3 scripts/EEG_Audit.py --subject 029
```

If the subject data live somewhere else:

```bash
python3 scripts/EEG_Audit.py --subjects-dir /path/to/subject_data
```

## What the audit does

1. Uses timestamps and actual `RAW_AF7`, `RAW_AF8` samples; it does **not** use Mind Monitor's precomputed `Alpha_AF7` or `Beta_AF7` values.
2. Reports observed sampling rates and rejects low-rate exports. Gaps are split; no signal interpolation occurs across gaps. Irregular timestamps are flagged.
3. Screens 4-second windows every 2 seconds, separately for both sensors. Obvious zero/1450 boundary-value occupancy (a heuristic), constant runs, missing values, or irregular timestamps cause hard failure.
4. For candidate windows, applies a 0.5–45 Hz Butterworth band-pass filter with zero-phase filtering. At 45 Hz low-pass, 60 Hz line noise is outside the analysis band, so no separate notch is necessary. Repeats the power calculation using 1–40 Hz as a sensitivity check. **Caution:** filtering these short windows can introduce edge effects; inspect the results before publication.
5. Calculates Welch periodogram densities and integrates 8–13 Hz alpha, 13–30 Hz beta, 20–30 Hz high beta, 1–4 Hz slow activity, and 35–45 Hz high-frequency activity. Ratios are beta / alpha.
6. Flags large excursions, abrupt jumps, extreme slow activity (possible ocular), and unusual high-frequency-to-beta levels (possible muscle). These are **warnings**, NOT a definitive diagnosis; two frontal sensors cannot reliably distinguish all neural and non-neural signals.
7. Summarizes **primary** (hard checks pass), **strict** (hard checks pass AND no warnings), and **paired** (both frontal sensors pass in the same window) results. Band ratio is **mean beta power / mean alpha power**, not mean of per-window ratios.

## Files produced

- `eeg_recording_qc.csv`: one row per input file, including exclusions and sampling estimates.
- `eeg_window_qc_and_power.csv`: AF7/AF8 window powers, flags, raw/alternate filtering ratio, and time bounds.
- `eeg_frontal_band_summary.csv`: channel-specific powers and ratios in primary/strict/paired variants.
- `eeg_paired_windows_inventory.csv`: common clean-window inventory per recording.
- `figures/sub###/*_audit.png`: original raw signal with hard-rejected intervals highlighted, and alpha/beta power trajectories. Warning-only intervals are **not shaded**.
- `audit_settings.json`: settings for reproducibility.

## Interpretation and limitations

**AUDITED means that processing completed, not that the recording is artifact-free.** Window-level hard flags are conservative automated heuristics with configurable thresholds and have not been validated against expert labels. Sensor units and the interpretation of the repeated boundary values (0 and 1450) need confirmation. Potentially contaminated windows may still pass; genuine signals may be flagged. The code does not perform validated blink removal, ICA, source separation, or electrode repair; the word "clean" here means passes automated screening.

**Timing:** Mind Monitor's reported timestamps may have quantization and jitter. The audit assumes that 256 consecutive raw EEG samples at approximately the intended cadence represent four seconds; it checks for major gaps and duration errors, but it cannot certify hardware-perfect timing. Out-of-order/duplicate timestamps are counted and removed without synthesizing samples.

**Movement:** AUX values and their meanings have not been independently confirmed, so there is no assumed quantitative accelerometer artifact rejection. Eye/muscle labels are only candidate warnings based on spectral characteristics.

**Overlap:** Four-second windows are offset two seconds, so the `accepted_window_seconds_overlapping` column is a *sum of overlapping window durations*, not distinct clean recording duration. Don't use this as unique seconds of clean EEG.

**Phase:** The recording filename gives a phase label, not an event timestamp. Phase labels do not establish when stress occurred. Pre/post comparisons require matching sessions and independent validation of when stress was induced.

**Statistics:** Treat participants as independent units, not hundreds of overlapping windows. Examine alpha, beta, and their ratio separately. Be attentive to large changes in AF7 vs AF8, and compare primary vs strict/filter-sensitivity variants. Large or phase-dependent differences should lead to visual inspection before inference.

**Validation still needing human work:** visually inspect traces and plots, validate sentinel values/units, review all borderline windows, verify protocol phase timing, and obtain independent artifact annotations where possible. This script provides the reproducible first audit, not proof of physiological validity.
