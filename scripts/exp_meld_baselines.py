import os, json, argparse, math, joblib
import numpy as np
import soundfile as sf
import torch

from qwen_omni_rep_eng.data.meld import load_meld
from qwen_omni_rep_eng.models.omni import OmniRunner
from qwen_omni_rep_eng.analysis.probing import proba_margin_joy_sad
from qwen_omni_rep_eng.constants import JOY, SAD
from qwen_omni_rep_eng.steering.thinker import register_delta_steer, make_delta, random_delta_like
from qwen_omni_rep_eng.metrics.audio import delta_f0_mean
from qwen_omni_rep_eng.metrics.text import token_overlap_f1, length_ratio
from qwen_omni_rep_eng.metrics.plotting import plot_alpha_sweep, plot_method_bars, plot_drift_vs_gain

def _compute_delta_from_train(runner, n=200, layer=24):
    tr = load_meld(split="train", with_av=False, limit=n)
    happy, sad = [], []
    for ex in tr:
        if ex.emotion.lower() not in ("joy","sadness"): continue
        hs, _ = runner.thinker_hidden_states(runner.make_inputs(runner.convo_from_text(ex.utterance), use_audio_in_video=False))
        feat = hs[layer][0, -1, :].detach().cpu().numpy()
        (happy if ex.emotion.lower()=="joy" else sad).append(feat)
    if len(happy) < 5 or len(sad) < 5:
        print("[warn] small happy/sad sample; delta may be weak.")
    return make_delta(happy, sad).to(runner.device), len(happy), len(sad)

def _affect_gain(runner, clf, layer, text_before, text_after) -> float:
    # Use Thinker features for the *generated* text only (proxy); if you prefer,
    # you can run features on the original transcript too and subtract.
    # Here: margin(after) - margin(before), where 'before' is with alpha=0 (null).
    def _margin(t): 
        hs,_ = runner.thinker_hidden_states(runner.make_inputs(runner.convo_from_text(t), use_audio_in_video=False))
        x = hs[layer][0,-1,:].detach().cpu().numpy()
        return proba_margin_joy_sad(clf, x, JOY, SAD)
    return _margin(text_after) - _margin(text_before)

def _gen_with_method(runner, video_path, method, layer, delta, alpha, seed=0):
    """Returns (text, audio_np)."""
    conv = runner.convo_from_video(video_path, prompt="Repeat the original speech content using the same words. Then speak it in a cheerful tone.")
    inputs = runner.make_inputs(conv, use_audio_in_video=True)

    layers = runner._iter_thinker_layers()
    span = slice(0, inputs["input_ids"].shape[1])

    handle = None
    if method == "null":
        pass
    elif method == "prompt_cheerful":
        # No hook; prompt carries the control signal
        pass
    elif method == "delta_bestlayer":
        handle = register_delta_steer(layers[layer], delta, span, alpha=alpha)
    elif method == "delta_early":
        early = max(4, layer//4)
        handle = register_delta_steer(layers[early], delta, span, alpha=alpha)
    elif method == "delta_random":
        rnd = random_delta_like(delta, seed=seed)
        handle = register_delta_steer(layers[layer], rnd, span, alpha=alpha)
    else:
        raise ValueError(method)

    try:
        text_ids, audio = runner.generate_any2any(
            inputs, thinker_do_sample=False, talker_do_sample=True
        )
    finally:
        if handle is not None: handle.remove()

    text = runner.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    y = audio.reshape(-1).detach().cpu().numpy()
    return text, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--probe", type=str, default="outputs/meld_probe.joblib")
    ap.add_argument("--meta",  type=str, default="outputs/meld_probe_meta.json")
    ap.add_argument("--n", type=int, default=20)             # number of MELD test clips
    ap.add_argument("--alpha_grid", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--out", type=str, default="outputs/meld_baselines.jsonl")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    clf = joblib.load(args.probe)
    meta = json.load(open(args.meta))
    best_layer = int(meta["best_layer"])

    runner = OmniRunner(model_name=args.model, thinker_only=False)

    # Prepare Δ for baselines
    delta, nh, ns = _compute_delta_from_train(runner, n=200, layer=best_layer)
    print(f"[delta] built from {nh} happy / {ns} sad examples at layer {best_layer}")

    # Pick test items with video paths
    test = [ex for ex in load_meld(split="test", with_av=True) if ex.video_path][:args.n]
    methods = ["null","prompt_cheerful","delta_random","delta_early","delta_bestlayer"]

    with open(args.out, "w") as fo:
        for ex in test:
            # Baseline text (null) for affect-gain reference
            t0, y0 = _gen_with_method(runner, ex.video_path, "null", best_layer, delta, alpha=0.0)
            for method in methods:
                for alpha in (args.alpha_grid if "delta" in method else [0.0]):  # only alpha sweep for delta methods
                    t, y = _gen_with_method(runner, ex.video_path, method, best_layer, delta, alpha=alpha)
                    # Metrics
                    affect_gain = _affect_gain(runner, clf, best_layer, t0, t)
                    content_f1  = token_overlap_f1(ex.utterance, t)
                    len_ratio   = length_ratio(ex.utterance, t)
                    df0         = delta_f0_mean(y0, y, sr=24000)
                    rec = {
                        "dialogue_id": ex.dialogue_id, "utterance_id": ex.utterance_id,
                        "label": ex.emotion, "method": method, "alpha": alpha,
                        "affect_gain": affect_gain, "content_f1": content_f1,
                        "length_ratio": len_ratio, "delta_f0_mean": df0
                    }
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    print(rec)

    # Plots
    plot_alpha_sweep(args.out, "outputs/alpha_sweep_joysad.png")
    plot_method_bars(args.out, "outputs/method_bars_f0_and_content.png")
    plot_drift_vs_gain(args.out, "outputs/scatter_drift_vs_gain.png")
    print("Saved: outputs/alpha_sweep_joysad.png, outputs/method_bars_f0_and_content.png, outputs/scatter_drift_vs_gain.png")

if __name__ == "__main__":
    main()
