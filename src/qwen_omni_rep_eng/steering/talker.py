from __future__ import annotations
from typing import Optional, Tuple
import torch

# We intercept near the Talker code predictor to nudge codec logits toward "happy".
# Exact module names may vary; our OmniRunner.register_thinker_hooks locates a module
# whose name contains 'code_predictor' and lets you register a forward hook there.
# Doc reference: Talker predicts discrete speech codecs (multi-codebook) autoregressively.

def register_talker_logit_bias(module: torch.nn.Module,
                               bias_fn) -> torch.utils.hooks.RemovableHandle:
    """
    bias_fn: callable taking 'out' and returning a new 'out' with modified logits.
    'out' can be a ModelOutput with .logits or a tuple whose last elt is logits.
    """
    def _hook(_m, _inp, out):
        return bias_fn(out)
    return module.register_forward_hook(_hook)

def make_happy_bias_fn(alpha: float = 0.5):
    """
    Example bias: add small positive bias to code indices correlated with "happy" prosody.
    In practice you'd learn or estimate this from Talker hidden states on MELD happy vs sad,
    but we expose a scaffold here: out.logits += alpha * B, where B is precomputed.
    """
    B = None  # Placeholder: load from a learned .pt file mapping codebooks->indices
    def bias(out):
        # Support ModelOutput or tuple
        if hasattr(out, "logits"):
            logits = out.logits
            if B is not None:
                out.logits = logits + alpha * B.to(logits.device)
            return out
        # tuple: assume last is logits
        if isinstance(out, (tuple, list)):
            *rest, logits = out
            if B is not None:
                logits = logits + alpha * B.to(logits.device)
            return (*rest, logits)
        return out
    return bias
