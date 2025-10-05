from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch

# We support SAELens >=3 in a thin wrapper; if not installed, raise informative error.
try:
    import sae_lens
    from sae_lens import SAE, SAETrainer, SAETrainingConfig
    SAE_AVAILABLE = True
except Exception:
    SAE_AVAILABLE = False
    SAE = object  # type: ignore

def train_sae_on_layer(acts: List[np.ndarray],
                       dict_mult: int = 8,
                       l1_coeff: float = 0.01) -> SAE:
    if not SAE_AVAILABLE:
        raise ImportError("sae-lens not installed. Install with `pip install sae-lens` or use extras 'interp'.")
    X = torch.tensor(np.stack(acts, axis=0))  # [N, d]
    d_model = X.shape[-1]
    config = SAETrainingConfig(
        model_dim=d_model, dict_size=d_model * dict_mult, l1_coefficient=l1_coeff
    )
    trainer = SAETrainer(config)
    sae = trainer.train(X)
    return sae  # exposes .encode() / .decode(), matching our steering util
