from pathlib import Path
import numpy as np
import pandas as pd
import os

def label_wesad_subject(
    subject_id: str,
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    tags_root: Path = Path("../data/WESAD/raw/WESAD_raw"),
    output_dir: Path = Path("../data/WESAD/derived"),
    min_segment_seconds: float = 45.0,
    df_fs: float = 4.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Label feature windows for a WESAD subject with baseline / stress / amusement.

    Parameters
    ----------
    subject_id : e.g. "S2"
    df : merged 4 Hz dataframe with a DatetimeIndex
    features_df : windowed features with 'start' and 'end' columns (datetimes)
    """
    # 1) Load labels from pickle
    tags_path = tags_root / subject_id / f"{subject_id}.pkl"
    if not tags_path.exists():
        raise FileNotFoundError(f"Pickle not found for {subject_id}: {tags_path}")

    tags_obj = pd.read_pickle(tags_path)
    if "label" not in tags_obj:
        raise KeyError(f"'label' not found in tags_obj for {subject_id}. Keys: {list(tags_obj.keys())}")

    lab = np.array(tags_obj["label"])
    n_lab = len(lab)
    n_df = len(df)

    if verbose:
        print(f"[{subject_id}] n_lab = {n_lab}, n_df = {n_df}")

    # 2) Collapse labels to {0,1,2} (baseline/stress/amusement)
    collapsed = np.full_like(lab, fill_value=-1)
    collapsed[lab == 0] = 0
    collapsed[lab == 1] = 1
    collapsed[lab == 2] = 2

    # Estimate original label freq from duration
    f_lab_est = n_lab * df_fs / max(n_df, 1)
    min_samples_lab = int(min_segment_seconds * f_lab_est)

    if verbose:
        print(f"[{subject_id}] f_lab_est ≈ {f_lab_est:.2f} Hz, min_samples_lab ≈ {min_samples_lab}")

    # 3) Stable segments in label index space
    change_points = np.where(np.diff(collapsed) != 0)[0] + 1
    segments = []
    start = 0
    for cp in change_points:
        segments.append((start, cp, collapsed[start]))
        start = cp
    segments.append((start, n_lab, collapsed[start]))

    segments = [s for s in segments if s[2] in (0, 1, 2)]
    stable = [s for s in segments if (s[1] - s[0]) >= min_samples_lab]
    stable = sorted(stable, key=lambda s: s[0])

    if verbose:
        print(f"[{subject_id}] Stable B/S/A segments: {len(stable)}")
        for seg in stable:
            print("  ", seg)

    if not any(s[2] == 1 for s in stable):
        raise ValueError(f"[{subject_id}] No stable stress segment found.")
    if not any(s[2] == 2 for s in stable):
        raise ValueError(f"[{subject_id}] No stable amusement segment found.")

    # 4) Pick key phase onsets
    b0_start = next(s[0] for s in stable if s[2] == 0)
    s1_start = next(s[0] for s in stable if s[2] == 1)
    a2_start = next(s[0] for s in stable if s[2] == 2)
    baseline_after = [s for s in stable if s[2] == 0 and s[0] > a2_start]
    b_last_start = baseline_after[-1][0] if baseline_after else b0_start

    key_lab_idxs = [b0_start, s1_start, a2_start, b_last_start]

    # 5) Map label indices → df indices
    def lab_idx_to_df_idx(i: int) -> int:
        return int(round(i * (n_df - 1) / max(n_lab - 1, 1)))

    key_df_idxs = [lab_idx_to_df_idx(i) for i in key_lab_idxs]
    key_df_idxs = [min(max(0, idx), n_df - 1) for idx in key_df_idxs]

    if verbose:
        print(f"[{subject_id}] key_lab_idxs: {key_lab_idxs}")
        print(f"[{subject_id}] key_df_idxs: {key_df_idxs}")

    tags = pd.Series(df.index[key_df_idxs]).sort_values().reset_index(drop=True)

    if verbose:
        print(f"[{subject_id}] len(tags) = {len(tags)}")
        print(f"[{subject_id}] tags:")
        print(tags)

    # 6) Build intervals + label names
    if len(tags) == 3:
        label_names = ["baseline", "stress", "amusement"]
    elif len(tags) == 4:
        label_names = ["baseline", "stress", "amusement", "baseline"]
    else:
        raise ValueError(f"[{subject_id}] Unexpected number of tag boundaries: {len(tags)}")

    intervals = list(zip(tags[:-1].to_list(), tags[1:].to_list()))
    if verbose:
        print(f"[{subject_id}] intervals:")
        for iv in intervals:
            print("  ", iv)

    def label_for_window(start, end):
        mid = start + (end - start) / 2
        for i, (s, e) in enumerate(intervals):
            if s <= mid < e:
                return label_names[i]
        if mid >= tags.iloc[-1]:
            return label_names[-1]
        return np.nan

    # 7) Apply labels to features_df
    out = features_df.copy()
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])

    out["label"] = out.apply(
        lambda r: label_for_window(r["start"], r["end"]),
        axis=1
    )

    label_map = {lab: i for i, lab in enumerate(sorted(set(label_names)))}
    out["label_code"] = out["label"].map(label_map)

    # 8) Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}_features_labeled.csv"
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[{subject_id}] Saved labeled features to {out_path}")
        print(f"[{subject_id}] Label counts:")
        print(out["label"].value_counts(dropna=False))

    return out
