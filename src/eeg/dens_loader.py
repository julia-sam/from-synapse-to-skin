from pathlib import Path
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DENS_RAW_ROOT = PROJECT_ROOT / "data" / "DENS" / "raw"


def load_dens_subject(subject_id: str):
    """
    Load raw EEG (.set/.fdt) and events .tsv for a DENS subject.
    """
    sub_dir = DENS_RAW_ROOT / subject_id
    eeg_dir = sub_dir / "eeg"

    print(f"DENS_RAW_ROOT = {DENS_RAW_ROOT}")
    print(f"Looking in: {eeg_dir}")

    set_files = list(eeg_dir.glob("*_task-Emotion_eeg.set"))
    if not set_files:
        raise FileNotFoundError(f"No .set file found in {eeg_dir}")
    set_path = set_files[0]

    print(f"[{subject_id}] Loading EEG from: {set_path}")
    raw = mne.io.read_raw_eeglab(set_path, preload=True)

    events_files = list(eeg_dir.glob("*_task-emotion_events.tsv"))
    if not events_files:
        raise FileNotFoundError(f"No events .tsv file found in {eeg_dir}")
    events_path = events_files[0]

    print(f"[{subject_id}] Loading events from: {events_path}")
    events_df = pd.read_csv(events_path, sep="\t")

    print(f"[{subject_id}] raw: {raw}, events rows: {len(events_df)}")
    return raw, events_df

def pick_event_label_column(events_df: pd.DataFrame) -> str:
    for col in ["label", "emotion", "condition", "stimulus", "trial_type"]:
        if col in events_df.columns:
            return col
    raise ValueError(
        f"Could not find a label/emotion column in events_df columns: {events_df.columns.tolist()}"
    )
