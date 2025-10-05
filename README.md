# Qwen-Omni-Rep-Eng

Representation engineering on Qwen3-Omni (Thinker–Talker MoE) for MELD:
- (a) Localize affect features (probing + activation patching)
- (b) Detect with linear probes and SAEs
- (c) Steer the model to shift sad → happy (text + prosody), preserving content

Backed by TransformerLens-style activation patching and SAE methodology.
Qwen3‑Omni usage follows Hugging Face's Qwen3OmniMoe* classes and processor.
MELD provides aligned text, audio, video with 7 emotion labels.

---

## Project Structure

```
omni-steer/
├── src/qwen_omni_rep_eng/
│   ├── models/omni.py         # OmniRunner wrapper for Qwen3-Omni
│   ├── data/meld.py           # MELD dataset loader
│   ├── analysis/
│   │   ├── probing.py         # Linear probes for emotion detection
│   │   ├── patching.py        # TransformerLens-style activation patching
│   │   └── sae.py             # Sparse autoencoder utilities
│   ├── steering/
│   │   ├── thinker.py         # Thinker layer delta-vector steering
│   │   └── talker.py          # Talker prosody modification
│   └── ui.py                  # Gradio UI for interactive demos
├── scripts/
│   ├── run_probe.py           # Layer sweep to find emotion localization
│   └── steer_demo.py          # End-to-end steering demo on video
├── MELD.Raw/                  # MELD dataset (download separately)
└── pyproject.toml             # Dependencies
```

---

## Quickstart

### 0) Environment
Make sure you have uv installed.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, install the dependencies.
```bash
# Install dependencies
uv pip install -e .
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

Download the MELD dataset:

```bash
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xzf MELD.Raw.tar.gz
```

This extracts the dataset into `MELD.Raw/` with the following structure:

```
MELD.Raw/
├── train_splits/              # Training video clips
├── output_repeated_splits_test/  # Test video clips
├── train_sent_emo.csv         # Training labels & metadata
├── dev_sent_emo.csv           # Development labels & metadata
└── test_sent_emo.csv          # Test labels & metadata
```

We use the HuggingFace datasets library to load labels/text, and read videos directly from MELD.Raw/.

### 2) Run probe localization

```bash
python scripts/run_probe.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --split train --layer_sweep 16 32 --limit 50
```

This sweeps across layers [16, 32) and trains a linear probe to detect emotions. Output shows macro-F1 per layer.

### 3) Steer a test clip

```bash
python scripts/steer_demo.py --video MELD.Raw/output_repeated_splits_test/dia263_utt3.mp4 --alpha 1.0
```

This:
- Extracts happy/sad features from a small training sample
- Computes a delta vector (happy - sad)
- Steers the model by adding `alpha * delta` to layer 24 activations
- Generates cheerful text + audio, saved to `steered.wav`

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
