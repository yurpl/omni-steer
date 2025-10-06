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
        feat = hs[layer][0, -1, :].detach().cpu().float().numpy()
        (happy if ex.emotion.lower()=="joy" else sad).append(feat)
    if len(happy) < 5 or len(sad) < 5:
        print("[warn] small happy/sad sample; delta may be weak.")
    return make_delta(happy, sad).to(runner.device), len(happy), len(sad)

def _affect_gain_same_text(runner, clf, layer, ref_text, method, delta, alpha):
    """Margin(after) - margin(base) on the SAME text, with correct class indexing."""
    conv_text = runner.convo_from_text(ref_text)
    inputs_text = runner.make_inputs(conv_text, use_audio_in_video=False)

    def _margin():
        hs, _ = runner.thinker_hidden_states(inputs_text)
        x = hs[layer][0, -1, :].detach().cpu().numpy()
        proba = clf.predict_proba(x[None, :])[0]
        # Map to indices in clf.classes_ (which are int labels 0..6 from training)
        classes = list(clf.classes_)
        joy_idx = classes.index(JOY)
        sad_idx = classes.index(SAD)
        return float(proba[joy_idx] - proba[sad_idx])

    base = _margin()
    if method not in ("delta_bestlayer", "delta_early", "delta_random"):
        return 0.0

    # Register steering and recompute margin
    layers = runner._iter_thinker_layers()
    span = slice(0, inputs_text["input_ids"].shape[1])
    handle = None
    try:
        if method == "delta_bestlayer":
            handle = register_delta_steer(layers[layer], delta.to(runner.device), span, alpha=alpha)
        elif method == "delta_early":
            early = max(4, layer // 4)
            handle = register_delta_steer(layers[early], delta.to(runner.device), span, alpha=alpha)
        elif method == "delta_random":
            rnd = random_delta_like(delta.to(runner.device), seed=0)
            handle = register_delta_steer(layers[layer], rnd, span, alpha=alpha)
        return _margin() - base
    finally:
        if handle is not None:
            handle.remove()

def _gen_with_method(runner, video_path, ref_text, method, layer, delta, alpha, seed=0, dialogue_id=None, utterance_id=None):
    """Returns (text, audio_np) while forcing the model to say exactly ref_text."""

    # Use sentinel tags + strict instruction
    sys_rules = (
        "You are a voice playback module.\n"
        "Rules:\n"
        "1) Output only the text BETWEEN <say> and </say>.\n"
        "2) Do not add or remove any words or punctuation.\n"
        "3) Respond in English only.\n"
    )
    neutral_prompt  = "Speak exactly the following text. Output only the text between the tags."
    cheerful_prompt = neutral_prompt + " Say it in a cheerful tone."

    prompt = cheerful_prompt if method == "prompt_cheerful" else neutral_prompt
    tagged = f"<say>{ref_text}</say>"

    # Build conversation with a system rule + the MELD video + the tagged transcript
    conv = [
        {"role": "system", "content": [{"type": "text", "text": sys_rules}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text",  "text": prompt},
            {"type": "text",  "text": tagged}
        ]}
    ]

    inputs = runner.make_inputs(conv, use_audio_in_video=True)

    # Steering hook (Thinker)
    layers = runner._iter_thinker_layers()
    span = slice(0, inputs["input_ids"].shape[1])
    handle = None
    if method == "delta_bestlayer":
        handle = register_delta_steer(layers[layer], delta, span, alpha=alpha)
    elif method == "delta_early":
        early = max(4, layer // 4)
        handle = register_delta_steer(layers[early], delta, span, alpha=alpha)
    elif method == "delta_random":
        rnd = random_delta_like(delta, seed=seed)
        handle = register_delta_steer(layers[layer], rnd, span, alpha=alpha)

    try:
        # Keep responses bounded and deterministic
        text_ids, audio = runner.generate_any2any(
            inputs,
            thinker_do_sample=False, talker_do_sample=True,
            max_new_tokens=min(128, max(32, len(ref_text.split()) * 4)),
            temperature=0.0
        )
    finally:
        if handle is not None:
            handle.remove()

    # Robust decode using runner helper (handles tensors / Generate outputs)
    raw = runner.decode_text(text_ids, inputs=inputs)[0].strip()

    # Extract between <say> ... </say> if model echoed tags
    def _between(s, a="<say>", b="</say>"):
        if a in s and b in s:
            return s.split(a, 1)[1].split(b, 1)[0].strip()
        return s
    text = _between(raw)
    
    # Debug: print decoded text for first few examples
    if hasattr(_gen_with_method, '_debug_count'):
        _gen_with_method._debug_count += 1
    else:
        _gen_with_method._debug_count = 1
    
    if _gen_with_method._debug_count <= 3:
        print(f"DECODED ({method}): {raw[:160]}")
        print(f"CLEAN ({method}): {text}")
        print(f"REF: {ref_text}")
        print("---")

    y = audio.reshape(-1).detach().cpu().numpy().astype("float32")
    
    # Save audio file for cheerful method
    if method == "prompt_cheerful" and dialogue_id is not None and utterance_id is not None:
        import soundfile as sf
        audio_filename = f"outputs/cheerful_dialogue_{dialogue_id}_utterance_{utterance_id}.wav"
        sf.write(audio_filename, y, 24000)
        print(f"Saved cheerful audio: {audio_filename}")
    
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

    # Pick test items with video paths (prefer sadness for sad->happy experiment)
    test_all = [ex for ex in load_meld(split="test", with_av=True) if ex.video_path]
    test = [ex for ex in test_all if ex.emotion.lower() == "sadness"][:args.n] or test_all[:args.n]
    methods = ["null","prompt_cheerful","delta_random","delta_early","delta_bestlayer"]

    with open(args.out, "w") as fo:
        for ex in test:
            # Null baseline audio/text with FIXED transcript
            t0, y0 = _gen_with_method(runner, ex.video_path, ex.utterance, "null", best_layer, delta, alpha=0.0, dialogue_id=ex.dialogue_id, utterance_id=ex.utterance_id)
            for method in methods:
                for alpha in (args.alpha_grid if "delta" in method else [0.0]):
                    t, y = _gen_with_method(runner, ex.video_path, ex.utterance, method, best_layer, delta, alpha=alpha, dialogue_id=ex.dialogue_id, utterance_id=ex.utterance_id)
                    affect_gain = _affect_gain_same_text(runner, clf, best_layer, ex.utterance, method, delta, alpha)

                    content_f1 = token_overlap_f1(ex.utterance, t)
                    len_ratio  = length_ratio(ex.utterance, t)
                    df0        = delta_f0_mean(y0, y, sr=24000)

                    rec = {
                        "dialogue_id": ex.dialogue_id, "utterance_id": ex.utterance_id,
                        "label": ex.emotion, "method": method, "alpha": alpha,
                        "affect_gain": affect_gain, "content_f1": content_f1,
                        "length_ratio": len_ratio, "delta_f0_mean": df0
                    }
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                    print(rec)

    # Plots
    plot_alpha_sweep(args.out, "outputs/alpha_sweep_joysad.png")
    plot_method_bars(args.out, "outputs/method_bars_f0_and_content.png")
    plot_drift_vs_gain(args.out, "outputs/scatter_drift_vs_gain.png")
    print("Saved: outputs/alpha_sweep_joysad.png, outputs/method_bars_f0_and_content.png, outputs/scatter_drift_vs_gain.png")

if __name__ == "__main__":
    main()
