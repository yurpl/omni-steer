from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch

def build_per_layer_probe(features: List[np.ndarray],
                          labels: List[int]) -> Tuple[LogisticRegression, float]:
    X = np.stack(features, axis=0)
    y = np.array(labels)
    clf = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, multi_class="ovr", n_jobs=None
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=None)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return clf, f1_score(y_test, preds, average="weighted")

def extract_token_feature(hidden_states: List[torch.Tensor],
                          layer: int, pos: int = -1) -> np.ndarray:
    """
    hidden_states: list of [B, T, d]
    return: [d]
    """
    H = hidden_states[layer]  # [B, T, d]
    h = H[0, pos, :].detach().cpu().float().numpy()
    return h

from typing import Optional, Tuple
import numpy as np

def proba_margin_joy_sad(clf, x: np.ndarray, joy_idx: int, sad_idx: int) -> float:
    """Return p(joy) - p(sadness) from a fitted sklearn LogisticRegression."""
    proba = clf.predict_proba(x[None, :])[0]  # [n_classes]
    return float(proba[joy_idx] - proba[sad_idx])

def batch_features_from_texts(runner, texts, layer: int, pos: int = -1) -> np.ndarray:
    """Utility to grab [N, d] features from Thinker hidden states for a list of texts."""
    feats = []
    for txt in texts:
        conv = runner.convo_from_text(txt)
        inputs = runner.make_inputs(conv, use_audio_in_video=False)
        hs, _ = runner.thinker_hidden_states(inputs)
        feats.append(hs[layer][0, pos, :].detach().cpu().numpy())
    return np.stack(feats, axis=0)
