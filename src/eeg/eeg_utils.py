import numpy as np
from mne.time_frequency import psd_array_welch

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}


def make_eeg_windows(raw, win_sec=4.0, step_sec=2.0):
    """
    Create sliding windows over the raw EEG.
    Returns a list of (start_sample, end_sample) pairs.
    """
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times

    win_samp = int(win_sec * sfreq)
    step_samp = int(step_sec * sfreq)

    starts = np.arange(0, n_samples - win_samp + 1, step_samp, dtype=int)
    windows = [(s, s + win_samp) for s in starts]

    print(f"Made {len(windows)} windows of {win_sec}s with step {step_sec}s")
    return windows


def bandpower_features(data, sfreq, ch_names):
    """
    Compute simple bandpower features from a window of EEG data.
    data: np.ndarray, shape (n_channels, n_samples)
    """
    psd, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=45, n_fft=256, average="mean"
    )  # psd shape: (n_channels, n_freqs)

    feats = {}
    for band_name, (fmin, fmax) in BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd[:, band_mask].mean(axis=1)  # per-channel
        feats[f"{band_name}_power_mean"] = float(band_power.mean())
        feats[f"{band_name}_power_std"] = float(band_power.std())

    # Frontal alpha asymmetry (F4 - F3), if available
    if "F3" in ch_names and "F4" in ch_names:
        f3_idx = ch_names.index("F3")
        f4_idx = ch_names.index("F4")
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = psd[:, alpha_mask].mean(axis=1)
        asym = float(alpha_power[f4_idx] - alpha_power[f3_idx])
        feats["alpha_asym_F4_minus_F3"] = asym
    else:
        feats["alpha_asym_F4_minus_F3"] = np.nan

    return feats