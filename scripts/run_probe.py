import argparse
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm

from qwen_omni_rep_eng.data.meld import load_meld
from qwen_omni_rep_eng.models.omni import OmniRunner
from qwen_omni_rep_eng.analysis.probing import build_per_layer_probe, extract_token_feature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--layer_sweep", nargs=2, type=int, default=[16, 32])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split("/")[-1]
    exp_id = f"{model_name}_{args.split}_L{args.layer_sweep[0]}-{args.layer_sweep[1]}_N{args.limit}_{timestamp}"
    
    print(f"Loading MELD {args.split} split with limit={args.limit}...")
    data = load_meld(split=args.split, with_av=False, limit=args.limit)
    print(f"Loaded {len(data)} examples")
    
    print(f"Initializing OmniRunner with model {args.model}...")
    runner = OmniRunner(model_name=args.model, thinker_only=True)
    print("Model loaded successfully")

    # Store results
    results = {
        "experiment_id": exp_id,
        "model": args.model,
        "split": args.split,
        "layer_sweep": args.layer_sweep,
        "limit": args.limit,
        "timestamp": timestamp,
        "layer_results": [],
        "best_layer": None,
        "best_wf1": None
    }
    
    best = (-1.0, None, None)  # (W-F1, layer, clf)
    for layer in tqdm(range(args.layer_sweep[0], args.layer_sweep[1]), desc="Layer sweep"):
        feats, labels = [], []
        for ex in tqdm(data, desc=f"Layer {layer}", leave=False):
            conv = runner.convo_from_text(ex.utterance)
            inputs = runner.make_inputs(conv, use_audio_in_video=False)
            hs, _ = runner.thinker_hidden_states(inputs)
            feats.append(extract_token_feature(hs, layer=layer, pos=-1))
            labels.append(["anger","disgust","sadness","joy","neutral","surprise","fear"].index(ex.emotion.lower()))
        clf, wf1 = build_per_layer_probe(feats, labels)
        tqdm.write(f"[Layer {layer:02d}] Weighted-F1={wf1:.3f}")
        
        # Store layer result
        results["layer_results"].append({
            "layer": layer,
            "weighted_f1": float(wf1)
        })
        
        if wf1 > best[0]: 
            best = (wf1, layer, clf)
            results["best_layer"] = layer
            results["best_wf1"] = float(wf1)

    print(f"BEST layer={best[1]} Weighted-F1={best[0]:.3f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{exp_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
