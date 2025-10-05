import torch
import numpy as np

def patch_activation(model, base_conv, src_conv, layer, span, run_cache_fn):
    cache_src = run_cache_fn(src_conv)
    A_src = cache_src[f"blocks.{layer}.resid_post"][0, span, :]
    
    def swap(m, inp, out):
        out = out.clone()
        out[0, span, :] = A_src
        return out
    
    thinker = model.thinker if hasattr(model, "thinker") else model
    handle = thinker.layers[layer].register_forward_hook(swap)
    inputs = {}  # Prepare base inputs
    outputs = model.generate(**inputs)  # Adjust for full gen
    handle.remove()
    return outputs

def steer_delta_vector(happy_data, sad_data, layer, alpha=1.0, pos=-1, run_cache_fn=None):
    mu_happy = np.mean([run_cache_fn(ex)[f"blocks.{layer}.resid_post"][0, pos].cpu().numpy() for ex in happy_data], axis=0)
    mu_sad = np.mean([run_cache_fn(ex)[f"blocks.{layer}.resid_post"][0, pos].cpu().numpy() for ex in sad_data], axis=0)
    delta = torch.tensor(mu_happy - mu_sad).to(next(iter(model.parameters())).device)
    
    def steer_hook(m, inp, out, span):
        out = out.clone()
        out[:, span, :] += alpha * delta
        return out
    
    return steer_hook  # Register: thinker.layers[layer].register_forward_hook(lambda m,i,o: steer_hook(m,i,o, span))

def sae_feature_steer(sae: SAE, features_to_boost, beta=1.0):
    def steer_fn(m, inp, out, span):
        out = out.clone()
        activations = sae.encode(out[:, span, :])  # [bs, seq, dict_size]
        for f in features_to_boost:
            activations[..., f] += beta * activations[..., f]  # Or set to target value
        steered = sae.decode(activations)
        out[:, span, :] += steered - out[:, span, :]  # Residual addition
        return out
    return steer_fn  # Register similarly