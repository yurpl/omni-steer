import os
from datasets import load_dataset
import moviepy.editor as mp
import soundfile as sf

def load_meld(split="train", with_av=True, assets_dir="assets/MELD.Raw"):
    ds = load_dataset("declare-lab/MELD", split=split)
    examples = []
    for ex in ds:
        item = {
            "utterance": ex["utterance"],
            "emotion": ex["emotion"],
            "dialogue_id": ex["dialogue_id"],
            "utterance_id": ex["utterance_id"]
        }
        if with_av:
            split_dir = f"{split}_split" if split != "validation" else "dev_splits_complete"
            video_path = os.path.join(assets_dir, split_dir, f"dia{item['dialogue_id']}_utt{item['utterance_id']}.mp4")
            audio_path = video_path.replace(".mp4", ".wav")
            if os.path.exists(video_path) and not os.path.exists(audio_path):
                clip = mp.VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path)
            item["video_path"] = video_path if os.path.exists(video_path) else None
            item["audio_path"] = audio_path if os.path.exists(audio_path) else None
        examples.append(item)
    return examples

def load_iemocap(split="train", with_av=True):
    raise NotImplementedError("IEMOCAP requires manual download and parsing from https://sail.usc.edu/iemocap/")