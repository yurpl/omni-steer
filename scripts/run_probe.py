import argparse, json, joblib
import os
from tqdm import tqdm
from qwen_omni_rep_eng.data.meld import load_meld
from qwen_omni_rep_eng.models.omni import OmniRunner
from qwen_omni_rep_eng.analysis.probing import build_per_layer_probe, extract_token_feature
from qwen_omni_rep_eng.constants import EMOTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--layer_sweep", nargs=2, type=int, default=[16, 32])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--out_probe", type=str, default="outputs/meld_probe.joblib")
    ap.add_argument("--out_meta",  type=str, default="outputs/meld_probe_meta.json")
    args = ap.parse_args()

    data = load_meld(split=args.split, with_av=False, limit=args.limit)
    runner = OmniRunner(model_name=args.model, thinker_only=True)

    best = (-1.0, None, None)
    for layer in tqdm(range(args.layer_sweep[0], args.layer_sweep[1]), desc="Layer sweep"):
        feats, labels = [], []
        for ex in tqdm(data, desc=f"Layer {layer}", leave=False):
            conv = runner.convo_from_text(ex.utterance)
            inputs = runner.make_inputs(conv, use_audio_in_video=False)
            hs, _ = runner.thinker_hidden_states(inputs)
            feats.append(extract_token_feature(hs, layer=layer, pos=-1))
            labels.append(EMOTIONS.index(ex.emotion.lower()))
        clf, f1 = build_per_layer_probe(feats, labels)
        tqdm.write(f"[Layer {layer:02d}] Weighted-F1={f1:.3f}")
        if f1 > best[0]: best = (f1, layer, clf)

    f1, layer, clf = best
    print(f"BEST layer={layer} F1={f1:.3f}")
    
    # Create outputs directory before saving files
    os.makedirs("outputs", exist_ok=True)
    
    joblib.dump(clf, args.out_probe)
    meta = {"best_layer": layer, "emotions": EMOTIONS, "weighted_f1": f1}
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()