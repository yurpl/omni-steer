import gradio as gr
from src.dataloaders import load_meld
from src.hooks import hook_qwen_omni
from src.trainers import train_emotion_probe
from src.utils import steer_delta_vector
import soundfile as sf
import os

# Preload (simplified; load checkpoints in practice)
model, proc, reg_hooks, run_cache = hook_qwen_omni()
meld_test = load_meld("test", with_av=True)
# probe = train_emotion_probe(...)  # Or load
# Assume best_layer = 20  # From localization

def steer_poc(video_path, alpha=1.0):
    if not video_path:
        return None, None
    conv = [{"role": "user", "content": [{"type": "video", "video": video_path}]}]
    inputs = proc.apply_chat_template(conv, use_audio_in_video=True, load_audio_from_video=True, return_tensors="pt").to(model.device)
    
    # Detect (placeholder)
    # cache = run_cache(conv)
    # feat = cache[f"blocks.20.resid_post"][0, -1]
    # emotion = probe.predict(feat.unsqueeze(0))
    
    # Steer
    steer_hook = steer_delta_vector([], [], 20, alpha)  # Pass happy/sad sets
    handle = model.thinker.layers[20].register_forward_hook(lambda m,i,o: steer_hook(m,i,o, slice(0, None)))
    text_ids, audio = model.generate(**inputs, thinker_do_sample=False, talker_do_sample=True)
    handle.remove()
    
    text = proc.batch_decode(text_ids)[0]
    audio_path = "output_happy.wav"
    sf.write(audio_path, audio.reshape(-1).cpu().numpy(), 24000)
    
    # Regeneration (placeholder; integrate SadTalker/EMO)
    video_out = video_path  # Replace with steered video
    
    return video_out, audio_path

demo = gr.Interface(
    fn=steer_poc,
    inputs=[gr.Video(label="Sad MELD Clip"), gr.Slider(0.5, 2.0, label="Alpha")],
    outputs=[gr.Video(label="Happy Video"), gr.Audio(label="Happy Audio")],
    title="Sad → Happy Steering POC"
)
demo.launch()