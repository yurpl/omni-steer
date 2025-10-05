import argparse
import soundfile as sf
import torch
from qwen_omni_rep_eng.models.omni import OmniRunner
from qwen_omni_rep_eng.steering.thinker import make_delta, register_delta_steer
from qwen_omni_rep_eng.data.meld import load_meld

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    runner = OmniRunner(model_name=args.model, thinker_only=False)
    # Quick delta from small MELD sample
    train = load_meld(split="train", with_av=False, limit=200)
    happy_feats, sad_feats = [], []
    layer = 24
    for ex in train:
        if ex.emotion.lower() not in ("joy", "sadness"): continue
        hs, _ = runner.thinker_hidden_states(runner.make_inputs(runner.convo_from_text(ex.utterance), use_audio_in_video=False))
        feat = hs[layer][0, -1, :].detach().cpu().float().numpy()
        (happy_feats if ex.emotion.lower() == "joy" else sad_feats).append(feat)
    delta = make_delta(happy_feats, sad_feats).to(runner.device)

    conv = runner.convo_from_video(args.video, prompt="Please describe this in a cheerful tone.")
    inputs = runner.make_inputs(conv, use_audio_in_video=True)
    layers = runner._iter_thinker_layers()
    span = slice(0, inputs["input_ids"].shape[1])
    handle = register_delta_steer(layers[layer], delta, span, alpha=args.alpha)
    try:
        text_ids, audio = runner.generate_any2any(inputs, thinker_do_sample=False, talker_do_sample=True)
    finally:
        handle.remove()

    text = runner.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("TEXT:", text)
    sf.write("steered.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    print("Saved steered.wav")

if __name__ == "__main__":
    main()
