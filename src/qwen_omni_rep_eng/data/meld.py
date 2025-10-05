import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# MELD text/labels via HF; AV via local MELD.Raw
# For AV filenames, MELD.Raw commonly uses dia{Dialogue_ID}_utt{Utterance_ID}.mp4 within split dirs.
# We robustly search by that pattern, but also allow user-provided path override.
# (MELD official README + AV bundle references.)  # Source: dataset docs & repo pages.

SPLIT_DIR = {
    "train": "train_splits",
    "validation": "dev_splits_complete",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

SPLIT_CSV = {
    "train": "train_sent_emo.csv",
    "validation": "dev_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

@dataclass
class MeldItem:
    utterance: str
    emotion: str
    dialogue_id: int
    utterance_id: int
    video_path: Optional[str]
    audio_path: Optional[str]

def _guess_split_dir(root: str, split: str) -> Optional[str]:
    # try common variants
    candidates = [
        SPLIT_DIR.get(split, split),
        "output_repeated_splits_test" if split == "test" else None,
    ]
    for c in candidates:
        if not c: continue
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return None

def _maybe_path(root_split: str, dialogue_id: int, utterance_id: int) -> Optional[str]:
    # common naming: dia{d}_utt{u}.mp4
    name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    p = os.path.join(root_split, name)
    if os.path.exists(p): return p
    # Some bundles have nested or alternative copies; scan small dir subset
    parent = root_split
    for fn in os.listdir(parent):
        if fn.startswith(f"dia{dialogue_id}_utt{utterance_id}") and fn.endswith(".mp4"):
            return os.path.join(parent, fn)
    return None

def load_meld(
    split: str = "train",
    with_av: bool = True,
    assets_dir: str = "MELD.Raw",
    limit: Optional[int] = None,
) -> List[MeldItem]:
    # Load from local CSV files
    csv_name = SPLIT_CSV.get(split, f"{split}_sent_emo.csv")
    csv_path = os.path.join(assets_dir, csv_name)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    items: List[MeldItem] = []

    root_split = None
    if with_av and os.path.isdir(assets_dir):
        root_split = _guess_split_dir(assets_dir, split)

    for idx, row in df.iterrows():
        if limit and idx >= limit: break
        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        video_path = None
        audio_path = None

        if with_av and root_split:
            vp = _maybe_path(root_split, dialogue_id, utterance_id)
            if vp:
                video_path = vp
                # audio will be extracted on demand by the processor (load_audio_from_video=True)

        items.append(MeldItem(
            utterance=row["Utterance"],
            emotion=row["Emotion"],
            dialogue_id=dialogue_id,
            utterance_id=utterance_id,
            video_path=video_path,
            audio_path=audio_path
        ))
    return items
