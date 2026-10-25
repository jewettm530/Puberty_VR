import mne

# 1. Load sample dataset built into MNE
# (This downloads example data automatically on its first run)
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"

# 2. Read the raw data into memory
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# 3. Clean the data with a bandpass filter (e.g., 1 to 40 Hz)
raw.filter(l_freq=1.0, h_freq=40.0)

# 4. Find time markers for events/stimuli
events = mne.find_events(raw, stim_channel="STI 014")

# 5. Segment the continuous data into 4-second "Epochs" around events
# (From 0.2 seconds before the event to 0.5 seconds after)
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True)

# 6. Plot the clean, segmented data
epochs.plot()
