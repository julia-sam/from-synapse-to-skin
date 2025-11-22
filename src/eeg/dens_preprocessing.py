from src.eeg.dens_loader import load_dens_subject, pick_event_label_column
from src.eeg.eeg_utils import make_eeg_windows, bandpower_features

import numpy as np
import pandas as pd
from pathlib import Path
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTISCALE_DIR = PROJECT_ROOT / "data" / "multiscale"

# Add your mapping here
EMOTION_TO_STATE = {
    'neutral_1_1': 'baseline',
    '12_2': 'amusement',
    '8_3': 'amusement',
    '4_4': 'stress',
    '15_5': 'stress',
    '17_6': 'baseline',
    '16_7': 'stress',
    'neutral_2_8': 'baseline',
    '7_9': 'baseline',
    '2_10': 'stress',
    '24_11': 'stress'
 }

def label_window_by_events(start_sec, end_sec, events_df, label_col, min_overlap_ratio=0.6):
    """
    Assign a label to a window [start_sec, end_sec] based on which event interval
    overlaps it the most. Require at least min_overlap_ratio of the window to be
    covered by that event; otherwise return None.
    """
    win_len = end_sec - start_sec
    if win_len <= 0:
        return None

    best_label = None
    best_overlap = 0.0

    for _, row in events_df.iterrows():
        ev_start = row["onset"]
        ev_end = row["onset"] + row.get("duration", 0.0)

        # overlap between [start_sec, end_sec] and [ev_start, ev_end]
        overlap_start = max(start_sec, ev_start)
        overlap_end = min(end_sec, ev_end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_label = row[label_col]

    if best_overlap / win_len >= min_overlap_ratio:
        return best_label
    return None

def build_dens_msff_for_subject(subject_id: str,
                                win_sec: float = 4.0,
                                step_sec: float = 2.0,
                                min_overlap_ratio: float = 0.6):
    raw, events_df = load_dens_subject(subject_id)
    sfreq = raw.info["sfreq"]
    ch_names = raw.info["ch_names"]

    label_col = pick_event_label_column(events_df)

    # --- Make sure event times are in seconds, not samples ---
    events_df = events_df.copy()
    rec_dur = raw.times[-1]
    max_onset = events_df["onset"].max()
    median_dur = events_df["duration"].median() if "duration" in events_df else np.nan

    # Heuristic: if event onsets are an order of magnitude above recording duration
    # and durations look like samples (>> rec_dur), convert to seconds.
    onset_ratio = max_onset / rec_dur if rec_dur else np.inf
    looks_like_samples = onset_ratio > 8 or (np.nan_to_num(median_dur, nan=0) > rec_dur * 2)
    if looks_like_samples:
        print(f"[{subject_id}] Converting event onsets/durations from samples to seconds")
        events_df["onset"] = events_df["onset"] / sfreq
        if "duration" in events_df:
            events_df["duration"] = events_df["duration"] / sfreq
    else:
        print(f"[{subject_id}] Treating event onsets/durations as seconds (max_onset={max_onset:.1f}, rec_dur={rec_dur:.1f})")

    # Keep only stimulation events (stm) for emotional labeling
    if "trial_type" in events_df.columns:
        events_df = events_df[events_df["trial_type"] == "stm"].copy()

    # If still empty, warn
    if len(events_df) == 0:
        print(f"[{subject_id}] WARNING: No 'stm' events found!")


    windows = make_eeg_windows(raw, win_sec=win_sec, step_sec=step_sec)
    data = raw.get_data(picks="eeg")  # (n_channels, n_samples)

    rows = []
    kept = 0
    skipped = 0

    for start_samp, end_samp in windows:
        start_sec = start_samp / sfreq
        end_sec = end_samp / sfreq

        raw_label = label_window_by_events(start_sec, end_sec, events_df, label_col, min_overlap_ratio)
        if raw_label is None:
            skipped += 1
            continue  # skip ambiguous windows

        # map to core state if mapping defined, otherwise keep raw_label
        state = EMOTION_TO_STATE.get(raw_label, raw_label)

        segment = data[:, start_samp:end_samp]
        feats = bandpower_features(segment, sfreq, ch_names)

        row = {
            "subject_id": subject_id,
            "layer": "brain",
            "start": start_sec,
            "end": end_sec,
            "raw_label": raw_label,
            "label": state,
        }
        row.update(feats)
        rows.append(row)
        kept += 1

    df = pd.DataFrame(rows)

    # Add numeric label_code for your core states, if you like
    if "label" in df.columns:
        unique_states = sorted(df["label"].dropna().unique())
        label_map = {lab: i for i, lab in enumerate(unique_states)}
        df["label_code"] = df["label"].map(label_map)

    label_counts = df["label"].value_counts(dropna=False).to_dict() if not df.empty else {}
    print(f"[{subject_id}] MSFF rows: {len(df)} (kept={kept}, skipped={skipped}, labels={label_counts})")
    return df

def build_all_dens_msff(subject_ids, save: bool = True, filename: str = "dens_brain.csv") -> pd.DataFrame:
    frames = []
    for sid in subject_ids:
        print(f"\n=== Processing {sid} ===")
        df_sid = build_dens_msff_for_subject(sid)
        frames.append(df_sid)

    # Combine all subjects
    all_df = pd.concat(frames, ignore_index=True)

    # Save if requested
    if save:
        MULTISCALE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MULTISCALE_DIR / filename
        all_df.to_csv(out_path, index=False)
        print(f"Saved combined DENS MSFF to {out_path}")

    return all_df
