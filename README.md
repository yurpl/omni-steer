# Qwen-Omni-Rep-Eng

Representation engineering on Qwen3-Omni (Thinker–Talker MoE) for MELD:
- (a) Localize affect features (probing + activation patching)
- (b) Detect with linear probes and SAEs
- (c) Steer the model to shift sad → happy (text + prosody), preserving content

Backed by TransformerLens-style activation patching and SAE methodology.
Qwen3‑Omni usage follows Hugging Face's Qwen3OmniMoe* classes and processor.
MELD provides aligned text, audio, video with 7 emotion labels.

---

## Quickstart

### 0) Environment
```bash
# Recommended: uv or venv
pip install -U pip
pip install uv
uv sync
```

> If transformers>=4.57.0 is not available in your env and you see missing classes like
> Qwen3OmniMoeForConditionalGeneration, install Transformers from source:
>
> ```bash
> pip install "git+https://github.com/huggingface/transformers"
> ```
>
> (The official Qwen3‑Omni doc page notes that "main" sometimes requires source install.)

### 1) Data: MELD

Download raw AV:

* Official bundle: MELD.Raw.tar.gz (see repo README). Unpack to assets/MELD.Raw/.

We'll also use Hugging Face's declare-lab/MELD for text/labels (robust splits).

Directory (after extract):

```
assets/MELD.Raw/
  train_split/
  dev_splits_complete/
  output_repeated_splits_test/
  test_split/               # sometimes present depending on bundle version
```

### 2) Run probe localization

```bash
python scripts/run_probe.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --split train --layer_sweep 12 36
```

### 3) Steer a test clip

```bash
python scripts/steer_demo.py --video assets/MELD.Raw/test_split/dia0_utt2.mp4 --alpha 1.2
```

### 4) UI (POC)

```bash
python src/qwen_omni_rep_eng/ui.py
```

---

## What you get

* Localization heatmaps (layer × token span) identifying where affect lives.
* Detectors: per-layer logistic probes; optional SAE features & dashboards.
* Steering:

  * Thinker Δ‑vector or SAE‑feature edits → text/internal affect.
  * Talker prosody nudge near code predictor → happy speech (pitch/energy/timing).

---

## Notes

* Classes & IO: Use Qwen3OmniMoeForConditionalGeneration + Qwen3OmniMoeProcessor. generate(...) returns (text_ids, audio) when audio output is enabled. Generation args prefixed with thinker_ vs talker_.
* Activation patching: Our patching.py mirrors the canonical pattern from TransformerLens tutorials.
* Prosody steering: We hook near the Talker code predictor (multi‑codebook AR codec) to nudge happy prosody while preserving words.
* Ethics: Any regenerated A/V is watermarked and labeled "Edited for research—emotion changed." Follow dataset license and use constraints.

---
