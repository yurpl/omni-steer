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
