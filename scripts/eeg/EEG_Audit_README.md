# EEG processing and audit

Run from the Puberty_VR project root.

```bash
python3 scripts/eeg/EEG_Audit.py
python3 scripts/eeg/process_all_eeg.py
python3 scripts/eeg/summarize_eeg.py
```

The master runner also calls these stages in order. Preprocessing owns the
sample selection, filtering, epoch rejection and spectral computations. Audit
uses those saved results. The collector aggregates existing spectral tables.

## Selection and fixed spectral settings

- Subjects 039 through 052, inclusive, use the second half of **each EEG
  recording**. The midpoint is half the original elapsed acquisition duration.
  Selection happens before filtering or epoch construction. Every phase,
  including baseline, pre-VR, post-VR and recovery, uses this rule.
- All other subjects use their complete recording.
- Original recording seconds, timestamps and CSV row positions are preserved.
- Sampling remains nominal 256 Hz, as in the supplied project.
- Welch uses a fixed 512-sample (2-second) periodic Hann taper, the original
  1024-point FFT and detrend=False with density scaling. The FFT is zero-padded
  from 512 samples and frequency bins are 0.25 Hz apart.
  Consecutive epochs start one second apart (50% overlap).
- Each accepted epoch contributes its periodogram to the arithmetic mean PSD.
  Averaging these identically tapered periodograms is the Welch estimate.
- At least one complete 512-sample epoch is required. Short recordings, short
  selected halves and short contiguous spans never use a smaller window.
  Trailing incomplete epochs are omitted.
- Filtering is the existing zero-phase 60 Hz notch, applied once to finite
  contiguous selected spans. Missing samples and timing gaps remain explicit.
- The original source offset is retained in spectral calculations. The signed
  offset-corrected samples are used for amplitude checks and manual review.
  The existing band edges are retained at theta 4–8, alpha 8–13, beta 13–30,
  gamma 30–45 and whole 4–45 Hz. Band power integrates PSD in µV²/Hz and has
  units µV². The stored log transform is log10(mean band power).
- The time-frequency PNG contains the accepted 2-second epoch PSDs for all available RAW channels.
  Each time cell is located at the epoch start and uses a one-second display
  width. Each estimate includes two seconds of data. Rejected epochs and
  missing spans are blank. Each colorbar uses its original automatic per-channel range.

## Voltage units and automatic exclusions

The Muse/Mind Monitor RAW values are treated as microvolts, with the 725 µV
offset documented in the previous viewer. Subtracting that fixed acquisition
offset yields signed voltage. Both the offset-corrected input and filtered
waveform are checked against -200 to +200 µV. One sample outside either bound
rejects the entire channel epoch. A value exactly at a bound is allowed.

Confirm this acquisition convention against your recordings. For already
signed µV exports, run:

```bash
python3 scripts/eeg/EEG_Audit.py --raw-offset-uv 0
```

The same setting is available to preprocessing and the notch comparison.
The amplitude setting is --amplitude-limit-uv and defaults to 200. Data in
volts must be converted to µV before this pipeline.

Additional automatic rejection checks cover missing samples, irregular epoch
duration, flatlines, long constant runs and the existing native Muse boundary
heuristic. The boundary heuristic is used only with the 725 µV offset setting.
Amplitude rejection flags large excursions. Manual inspection addresses
remaining contamination, including lower-amplitude muscle activity.

## Saved outputs

For every eligible recording:

- data/processed/eeg/sub###/<recording>_Preprocessed.npz
  contains selected offset-corrected input, filtered EEG, original elapsed
  times, source timestamps, source row positions, spans and configuration.
  The sample timeline remains intact after rejection.
- The adjacent <recording>_Preprocessed_WindowQC.csv contains every complete
  frontal channel epoch, its amplitude extrema, rejection reasons and acceptance.
- outputs/eeg/bandpower/sub_###/<recording>_PSD.csv contains mean accepted PSD
  by channel and frequency, in µV²/Hz, with accepted epoch counts.
- <recording>_FrontalLogPower.csv retains the six frontal band rows and adds
  selection and processing identity. A channel with no retained epochs has
  missing power and n_windows=0.
- <recording>_Result.csv retains the whitespace-delimited six ratio rows,
  with all available RAW channels in their original CSV column order.
- <recording>_WindowQC.csv duplicates the epoch ledger beside the spectra.
- <recording>_Processing.json records settings, crop boundaries, units,
  preprocessing identity, applied manual intervals and analysis identity.
- <recording>_PSD.png restores the original 2-by-4 figure. Top row is raw EEG
  PSD (all RAW channels overlaid), notch-filtered EEG PSD, movement PSD and
  movement magnitude. Bottom row is one spectrogram per RAW channel in CSV
  column order. TP9, AF7, AF8, TP10 and AUX are retained when available.
  A plain audit run exports this figure automatically. The frontal after-notch
  curves match _PSD.csv. Each channel uses its accepted epochs.
- <recording>_Spectrogram.png is exported by a plain audit run.
- The --preprocessing-figures option retains the supplied default of enabling
  a TIFF export. The TIFF shows the same original eight-panel diagnostic layout.
  Use --no-preprocessing-figures to omit TIFF files.

Audit continues to write its recording, epoch, review-priority, threshold,
candidate-power and pre/post comparison reports under outputs/eeg/audit/.
It also writes an audit overview per recording and optional review galleries.

Primary audit powers and collected band powers use the same accepted
epochs per channel. Audit strict results additionally exclude diagnostic
warnings. BOTH/paired estimates require AF7 and AF8 to pass in the same
epoch. Pooled beta/alpha is the ratio of mean powers. The legacy ratio
matrix stores the mean of epoch ratios.

Do not sum overlapping epoch durations as independent data. Unique retained
coverage is available in the audit summaries. Participants remain the unit
for inferential tests.

## Manual review after preprocessing

Run the audit first, then open the saved dataset:

```bash
python3 scripts/eeg/EEG_Manual_Review.py --subject 039 --phase pre_vr
```

The reviewer reads the saved NPZ and epoch ledger. It shows the filtered
voltage and automatic rejected spans. Press A to mark BAD_blink, BAD_muscle,
BAD_contact or a custom BAD_ label. REVIEW_ labels are notes.

For one channel use --channel AF7 or --channel AF8. With BOTH or ALL, GUI
annotations apply to both loaded frontal channels. Close the window to save
FIF, interval CSV and annotated waveform PDF files.

Rerun the audit and collector after review to exclude every epoch overlapping
a matching human BAD_ interval. Automatic labels start BAD_auto_ and are
reconstructed from current QC rather than reapplied as human exclusions.

The preprocessing ID binds annotations to the source samples, selected
period and preprocessing settings. The analysis ID also includes the applied
human exclusions. Recording approvals revert to PENDING if that analysis ID
changes. Existing reviewer notes are retained.

Legacy raw-based reviews and annotations from another preprocessing ID are
listed as ignored_annotation_files in metadata. Review the saved current
preprocessing before making new labels. Their different coordinate systems
are not mapped silently.

## Updating an existing project

Replace all EEG Python files from this update together and rerun the audit and
collector before using old cohort tables. Older exports fail the collector's
pipeline-version check. Existing raw-based annotation files are retained as
review records. Check affected old figures and reports when regenerating.

Dependencies are numpy, pandas, scipy and matplotlib. Install mne for manual
review. The scripts archive contains code only, so regression checks use
synthetic CSV/ZIP recordings. Run the included checks with:

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```
