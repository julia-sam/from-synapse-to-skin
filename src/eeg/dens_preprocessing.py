from src.eeg.dens_loader import load_dens_subject, pick_event_label_column
from src.eeg.eeg_utils import make_eeg_windows, bandpower_features

import numpy as np
import pandas as pd
from pathlib import Path
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTISCALE_DIR = PROJECT_ROOT / "data" / "multiscale"

# mapping
EMOTION_TO_STATE = {
    '12_10': 'amusement',
    '12_2': 'amusement',
    '12_3': 'amusement',
    '12_34': 'amusement',
    '12_4': 'amusement',
    '12_7': 'amusement',
    '12_8': 'amusement',
    '12_9': 'baseline',
    '13_10': 'amusement',
    '13_11': 'baseline',
    '13_2': 'baseline',
    '13_3': 'amusement',
    '13_4': 'baseline',
    '13_5': 'baseline',
    '13_6': 'amusement',
    '15_10': 'stress',
    '15_2': 'amusement',
    '15_3': 'stress',
    '15_32': 'baseline',
    '15_4': 'baseline',
    '15_5': 'baseline',
    '15_6': 'baseline',
    '15_7': 'stress',
    '15_8': 'amusement',
    '15_9': 'baseline',
    '16_10': 'baseline',
    '16_11': 'stress',
    '16_2': 'stress',
    '16_3': 'baseline',
    '16_36': 'stress',
    '16_5': 'stress',
    '16_6': 'stress',
    '16_7': 'stress',
    '16_9': 'stress',
    '17_10': 'baseline',
    '17_11': 'baseline',
    '17_2': 'stress',
    '17_5': 'stress',
    '17_6': 'baseline',
    '17_7': 'baseline',
    '17_8': 'stress',
    '17_9': 'stress',
    '1_10': 'baseline',
    '1_11': 'baseline',
    '1_2': 'baseline',
    '1_4': 'stress',
    '1_7': 'stress',
    '1_8': 'stress',
    '1_9': 'stress',
    '23_11': 'amusement',
    '23_3': 'baseline',
    '23_8': 'baseline',
    '24_10': 'baseline',
    '24_11': 'stress',
    '24_2': 'stress',
    '24_3': 'baseline',
    '24_4': 'baseline',
    '24_5': 'stress',
    '24_8': 'stress',
    '24_9': 'baseline',
    '27_11': 'baseline',
    '27_2': 'baseline',
    '27_3': 'baseline',
    '27_30': 'stress',
    '27_4': 'baseline',
    '27_5': 'baseline',
    '27_6': 'baseline',
    '27_7': 'baseline',
    '27_8': 'baseline',
    '28_3': 'baseline',
    '28_9': 'baseline',
    '2_10': 'stress',
    '2_11': 'baseline',
    '2_2': 'baseline',
    '2_27': 'stress',
    '2_4': 'stress',
    '2_5': 'stress',
    '2_6': 'baseline',
    '2_7': 'stress',
    '2_8': 'stress',
    '2_9': 'baseline',
    '3_11': 'baseline',
    '3_2': 'baseline',
    '3_3': 'baseline',
    '3_6': 'baseline',
    '3_7': 'baseline',
    '3_8': 'baseline',
    '3_9': 'amusement',
    '4_10': 'baseline',
    '4_11': 'stress',
    '4_2': 'baseline',
    '4_3': 'stress',
    '4_33': 'stress',
    '4_4': 'stress',
    '4_5': 'stress',
    '4_6': 'baseline',
    '4_7': 'baseline',
    '4_8': 'stress',
    '4_9': 'baseline',
    '5_10': 'baseline',
    '5_11': 'baseline',
    '5_2': 'stress',
    '5_3': 'baseline',
    '5_35': 'stress',
    '5_4': 'stress',
    '5_5': 'stress',
    '5_6': 'baseline',
    '6_10': 'amusement',
    '6_29': 'baseline',
    '6_5': 'baseline',
    '7_10': 'stress',
    '7_11': 'baseline',
    '7_2': 'amusement',
    '7_28': 'stress',
    '7_3': 'baseline',
    '7_4': 'stress',
    '7_6': 'baseline',
    '7_7': 'stress',
    '7_8': 'stress',
    '7_9': 'baseline',
    '8_10': 'amusement',
    '8_11': 'amusement',
    '8_2': 'baseline',
    '8_3': 'amusement',
    '8_4': 'baseline',
    '8_5': 'amusement',
    '8_7': 'amusement',
    '8_9': 'amusement',
    '9_10': 'amusement',
    '9_11': 'amusement',
    '9_2': 'baseline',
    '9_3': 'amusement',
    '9_4': 'amusement',
    '9_5': 'baseline',
    '9_6': 'amusement',
    '9_7': 'baseline',
    '9_9': 'amusement',
    'neutral_1_1': 'baseline',
    'neutral_2_31': 'baseline',
    'neutral_2_4': 'baseline',
    'neutral_2_5': 'baseline',
    'neutral_2_6': 'baseline',
    'neutral_2_7': 'baseline',
    'neutral_2_8': 'baseline',
    'neutral_2_9': 'baseline',
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
    failed_subjects = []
    
    for sid in subject_ids:
        print(f"\n=== Processing {sid} ===")
        try:
            df_sid = build_dens_msff_for_subject(sid)
            if len(df_sid) > 0:
                frames.append(df_sid)
                print(f"✓ {sid}: {len(df_sid)} rows added")
            else:
                print(f"⚠ {sid}: no valid windows extracted")
        except FileNotFoundError as e:
            print(f"✗ {sid}: FileNotFoundError — {e}")
            failed_subjects.append(sid)
        except Exception as e:
            print(f"✗ {sid}: {type(e).__name__} — {e}")
            failed_subjects.append(sid)

    if not frames:
        print("ERROR: No valid data extracted from any subject!")
        return pd.DataFrame()

    # Combine all subjects
    all_df = pd.concat(frames, ignore_index=True)

    # Save if requested
    if save:
        MULTISCALE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MULTISCALE_DIR / filename
        all_df.to_csv(out_path, index=False)
        print(f"\n✓ Saved combined DENS MSFF to {out_path}")

    print(f"\n=== Summary ===")
    print(f"Processed: {len(subject_ids)}")
    print(f"Succeeded: {len(subject_ids) - len(failed_subjects)}")
    print(f"Failed: {failed_subjects}")

    return all_df

def validate_dens_msff(df: pd.DataFrame) -> dict:
    """
    Quick validation checks on DENS MSFF dataset before training.
    Returns summary dict.
    """
    summary = {
        "n_rows": len(df),
        "n_subjects": df["subject_id"].nunique() if "subject_id" in df.columns else np.nan,
        "label_dist": df["label"].value_counts(dropna=False).to_dict() if "label" in df.columns else {},
        "n_features": len([c for c in df.columns if c not in ["subject_id", "layer", "start", "end", "raw_label", "label", "label_code"]]),
        "n_nans_per_col": df.isna().sum().to_dict(),
    }
    return summary
