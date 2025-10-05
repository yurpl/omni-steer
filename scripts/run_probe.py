import argparse
import numpy as np
from tqdm import tqdm

from qwen_omni_rep_eng.data.meld import load_meld
from qwen_omni_rep_eng.models.omni import OmniRunner
from qwen_omni_rep_eng.analysis.probing import build_per_layer_probe, extract_token_feature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--layer_sweep", nargs=2, type=int, default=[16, 32])
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    print(f"Loading MELD {args.split} split with limit={args.limit}...")
    data = load_meld(split=args.split, with_av=False, limit=args.limit)
    print(f"Loaded {len(data)} examples")
    
    print(f"Initializing OmniRunner with model {args.model}...")
    runner = OmniRunner(model_name=args.model, thinker_only=True)
    print("Model loaded successfully")

    best = (-1.0, None, None)  # (F1, layer, clf)
    for layer in tqdm(range(args.layer_sweep[0], args.layer_sweep[1]), desc="Layer sweep"):
        feats, labels = [], []
        for ex in tqdm(data, desc=f"Layer {layer}", leave=False):
            conv = runner.convo_from_text(ex.utterance)
            inputs = runner.make_inputs(conv, use_audio_in_video=False)
            hs, _ = runner.thinker_hidden_states(inputs)
            feats.append(extract_token_feature(hs, layer=layer, pos=-1))
            labels.append(["anger","disgust","sadness","joy","neutral","surprise","fear"].index(ex.emotion.lower()))
        clf, f1 = build_per_layer_probe(feats, labels)
        tqdm.write(f"[Layer {layer:02d}] Macro-F1={f1:.3f}")
        if f1 > best[0]: best = (f1, layer, clf)

    print(f"BEST layer={best[1]} F1={best[0]:.3f}")

if __name__ == "__main__":
    main()
