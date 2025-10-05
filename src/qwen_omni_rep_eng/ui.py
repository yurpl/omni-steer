import os
import gradio as gr
import soundfile as sf
import torch

from .data.meld import load_meld
from .models.omni import OmniRunner
from .steering.thinker import make_delta, register_delta_steer

# Minimal demo: steer Thinker; (optional) Talker bias could be added similarly.
runner = OmniRunner(model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct", thinker_only=False)

# (Optional) Precomputed delta (load from disk in practice)
_precomputed = {"layer": 24, "alpha": 1.1, "delta": None}

def _compute_or_load_delta():
    if _precomputed["delta"] is not None:
        return _precomputed["delta"], _precomputed["layer"]
    # Fallback: compute small delta from a tiny sample (text-only proxy)
    train = load_meld(split="train", with_av=False, limit=200)
    happy_feats, sad_feats = [], []
    layer = _precomputed["layer"]
    for ex in train:
        conv = runner.convo_from_text(ex.utterance)
        inputs = runner.make_inputs(conv, use_audio_in_video=False)
        hs, _ = runner.thinker_hidden_states(inputs)
        feat = hs[layer][0, -1, :].detach().cpu().numpy()
        (happy_feats if ex.emotion.lower() in ("joy",) else
         sad_feats if ex.emotion.lower() in ("sadness",) else None)
        if ex.emotion.lower() == "joy":     happy_feats.append(feat)
        if ex.emotion.lower() == "sadness": sad_feats.append(feat)
    if len(happy_feats) < 5 or len(sad_feats) < 5:
        # Not enough data; use zeros (no-op)
        _precomputed["delta"] = torch.zeros_like(hs[layer][0, -1, :])
    else:
        _precomputed["delta"] = make_delta(happy_feats, sad_feats).to(runner.device)
    return _precomputed["delta"], layer

def steer_poc(video_path: str, alpha: float = 1.0, spk: str = "Chelsie"):
    if not video_path or not os.path.exists(video_path):
        return None, None, "Please provide a MELD video path."

    # Build conversation: video only
    conv = runner.convo_from_video(video_path, prompt="Please describe the scene.")
    inputs = runner.make_inputs(conv, use_audio_in_video=True)

    # Register Thinker delta steering on the whole sequence
    delta, layer = _compute_or_load_delta()
    layers = runner._iter_thinker_layers()
    span = slice(0, inputs["input_ids"].shape[1])  # entire sequence tokens
    handle = register_delta_steer(layers[layer], delta, span, alpha=alpha)

    try:
        text_ids, audio = runner.generate_any2any(
            inputs, thinker_do_sample=False, talker_do_sample=True, spk=spk
        )
    finally:
        handle.remove()

    text = runner.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    audio_path = "output_happy.wav"
    sf.write(audio_path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    # For now we return original video (hooking EMO/SadTalker is optional postprocess)
    return video_path, audio_path, text

with gr.Blocks(title="Sad → Happy Steering POC") as demo:
    gr.Markdown("### Qwen‑3‑Omni: Thinker/Talker steering (sad → happy)")
    with gr.Row():
        inp_video = gr.Video(label="MELD clip (sad)")
        alpha = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Steer strength (alpha)")
        spk = gr.Dropdown(["Chelsie", "Ethan"], value="Chelsie", label="Voice (Talker)")
    btn = gr.Button("Make it happy")
    out_video = gr.Video(label="Video (identity preserved)")
    out_audio = gr.Audio(label="Happy audio")
    out_text = gr.Textbox(label="Generated text")
    btn.click(steer_poc, [inp_video, alpha, spk], [out_video, out_audio, out_text])

if __name__ == "__main__":
    demo.launch()
