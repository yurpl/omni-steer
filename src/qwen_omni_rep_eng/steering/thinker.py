from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np
import torch

def make_delta(happy_feats: List[np.ndarray], sad_feats: List[np.ndarray]) -> torch.Tensor:
    mu_h = torch.tensor(np.mean(happy_feats, axis=0), dtype=torch.float32)
    mu_s = torch.tensor(np.mean(sad_feats,  axis=0), dtype=torch.float32)
    return (mu_h - mu_s)

def register_delta_steer(layer_module: torch.nn.Module,
                         delta: torch.Tensor,
                         span: slice,
                         alpha: float = 1.0):
    delta = delta.to(next(layer_module.parameters()).device)
    def _hook(_m, _inp, out):
        out = out.clone()
        out[:, span, :] += alpha * delta
        return out
    return layer_module.register_forward_hook(_hook)

def register_sae_feature_steer(layer_module: torch.nn.Module,
                               sae,  # SAELens SAE
                               feature_indices: Iterable[int],
                               span: slice,
                               beta: float = 1.0):
    # Encourages selected features by proportionally increasing them, then re-decoding delta
    device = next(layer_module.parameters()).device
    feature_indices = list(feature_indices)

    def _hook(_m, _inp, out):
        out = out.clone()
        hs = out[:, span, :]  # [B, Tspan, d]
        z = sae.encode(hs)    # [B, Tspan, k]
        for f in feature_indices:
            z[..., f] = z[..., f] + beta * torch.abs(z[..., f])
        steered = sae.decode(z)  # [B, Tspan, d]
        out[:, span, :] = steered
        return out
    return layer_module.register_forward_hook(_hook)

def delta_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x)

def random_delta_like(delta_ref: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Return a random vector with the SAME norm as delta_ref (control baseline)."""
    g = torch.Generator(device=delta_ref.device).manual_seed(seed)
    rnd = torch.randn(delta_ref.shape, device=delta_ref.device, dtype=delta_ref.dtype, generator=g)
    rnd = rnd / (torch.linalg.norm(rnd) + 1e-8) * (torch.linalg.norm(delta_ref) + 1e-8)
    return rnd
