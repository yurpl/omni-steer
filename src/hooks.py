import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

def hook_qwen_omni(model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct", use_thinker_only=False):
    if use_thinker_only:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    else:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
    
    def register_hooks(model, cache={}):
        handles = []
        thinker = model.thinker if hasattr(model, "thinker") else model
        for i, layer in enumerate(thinker.layers):
            def save_resid_post(idx):
                def hook(m, inp, out):
                    cache[f"blocks.{idx}.resid_post"] = out[0].detach()  # hidden_states
                return hook
            handles.append(layer.register_forward_hook(save_resid_post(i)))
            # Attn out
            def save_attn_out(idx):
                def hook(m, inp, out):
                    cache[f"blocks.{idx}.attn_out"] = out.detach()
                return hook
            handles.append(layer.self_attn.register_forward_hook(save_attn_out(i)))
            # MLP out
            def save_mlp_out(idx):
                def hook(m, inp, out):
                    cache[f"blocks.{idx}.mlp_out"] = out.detach()
                return hook
            handles.append(layer.mlp.register_forward_hook(save_mlp_out(i)))
        # Talker hook (if full model)
        if hasattr(model, "talker"):
            def save_talker_logits(m, inp, out):
                cache["talker.logits"] = out.detach()
            handles.append(model.talker.code_predictor.register_forward_hook(save_talker_logits))  # Adjust based on exact attr
        return cache, handles
    
    def run_and_cache(conversation, model, processor, cache, handles, use_audio_in_video=True):
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            use_audio_in_video=use_audio_in_video, load_audio_from_video=use_audio_in_video
        ).to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, thinker_do_sample=False)  # Minimal to fill cache
        for h in handles:
            h.remove()
        return cache
    
    return model, processor, register_hooks, run_and_cache