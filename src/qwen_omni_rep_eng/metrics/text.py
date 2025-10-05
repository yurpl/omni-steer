from __future__ import annotations
import re
from collections import Counter
import numpy as np

def _tok(s: str):
    return re.findall(r"[a-zA-Z']+", s.lower())

def token_overlap_f1(ref: str, hyp: str) -> float:
    """Cheap content-preservation proxy (no ASR needed)."""
    r, h = Counter(_tok(ref)), Counter(_tok(hyp))
    inter = sum((r & h).values())
    prec  = inter / max(sum(h.values()), 1)
    rec   = inter / max(sum(r.values()), 1)
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def length_ratio(ref: str, hyp: str) -> float:
    r = len(_tok(ref)); h = len(_tok(hyp))
    return float(h / max(r, 1))
