from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Optional
import torch

# Canonical activation patching pattern:
# - Run "base" and "source" inputs
# - During base forward, replace a target activation with the source activation and
#   measure change in an eval scalar (e.g., probe score).
# Mirrors TL tutorials, adapted to arbitrary module hooks.  # Ref: TL docs. 

def patch_resid_span(layer_module: torch.nn.Module,
                     source_value: torch.Tensor,
                     span: slice) -> Callable:
    """
    Returns a forward hook that replaces the given span in the layer's output with source_value.
    Assumes layer output is [B, T, d]; source_value is [T_span, d] (or broadcastable).
    """
    def _hook(_m, _inp, out):
        out = out.clone()
        out[:, span, :] = source_value
        return out
    return _hook

@torch.no_grad()
def activation_patching_single(
    model: torch.nn.Module,
    thinker_layers: List[torch.nn.Module],
    inputs_base: Dict[str, torch.Tensor],
    inputs_src: Dict[str, torch.Tensor],
    layer_idx: int,
    span: slice,
    get_eval_scalar: Callable[[Dict[str, torch.Tensor]], float],
) -> float:
    """
    Patch residual at one (layer_idx, span) from src to base and return delta eval.
    `get_eval_scalar` should run a forward pass and compute the scalar (e.g., probe score).
    """
    # Obtain source act
    out_src = model(**inputs_src, output_hidden_states=True, return_dict=True)
    resid_src = out_src.hidden_states[layer_idx][0, span, :].detach()

    handle = thinker_layers[layer_idx].register_forward_hook(patch_resid_span(thinker_layers[layer_idx],
                                                                              resid_src, span))
    try:
        base_before = get_eval_scalar(inputs_base)
        base_after = get_eval_scalar(inputs_base)
        return base_after - base_before
    finally:
        handle.remove()
