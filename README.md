# Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting

This repository provides the reference implementation accompanying the paper  
**“Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting.”**

The codebase includes several deep learning baselines, a classical ARIMA reference model, and the proposed **two-stage patch-based forecasting pipeline** that combines convolutional tokenization with Transformer-based sequence modeling.

---

## Problem Setup

We consider supervised regression on multivariate time-series data.

### Data representation

- **Inputs**: `X ∈ R^{N×T×F}`
  - `N`: number of sequences (samples)
  - `T`: sequence length (timesteps)
  - `F`: number of input features

- **Targets**: `Y ∈ R^{N×T}`
  - One scalar regression target per timestep.

Datasets are stored as PyTorch dictionaries:

```text
{"X": X, "Y": Y}
```

---

## Proposed Method: Patch-Level Tokenization + Patch-Level Forecasting

The proposed method operates on **non-overlapping temporal patches** rather than on individual timesteps.

Given patch size `P`, each `T`-length sequence is segmented into:

- `K = floor(T / P)` patches
- effective length `T_eff = K * P` (the remainder, if any, is discarded)

### Stage A — CNN patch tokenizer (autoencoder pretraining)

A convolutional token encoder maps each patch to a `D`-dimensional embedding (token):

- tokens: `Z ∈ R^{N×K×D}`

In the current code, the tokenizer is trained as an **autoencoder**:

- encoder: `X → Z`
- decoder: `Z → reconstructed patches` with shape `R^{N×K×P×F}`

After encoding, a lightweight **token-level self-attention** block refines patch representations before decoding.

### Patch-level targets

For training and evaluation at patch resolution, timestep targets are reduced to patch targets:

- `Y_patch ∈ R^{N×K}`

The current implementation uses a configurable reduction (e.g., patch mean or patch last timestep). In the provided scripts, patch mean is used.

### Stage B — token export

The trained CNN encoder is used to export tokens (`Z`) for all sequences, along with the original timestep targets (`Y`) and metadata (including patch size).

### Stage C — QuantFormer on tokens (patch-level forecasting)

The Transformer model consumes token sequences and predicts **one scalar per patch**, i.e. it outputs `Ŷ_patch ∈ R^{N×K}`.

To perform forecasting rather than reconstruction, the code uses a rolling patch shift:

- inputs: `Z[:, :K-h, :]`
- targets: `Y_patch[:, h:]`

where `h` is the forecast horizon in **patches** (default `h=1`).

---

## Repository Structure

```text
Paper/
├── dataset/
│   ├── data.py
│   ├── paper_dataset.pt
│   └── paper_new_test.pt
│
├── baselineGRU/
├── baselineLSTM/
├── baselineTransModel/
├── baselineQuantTransModel/
├── arima/
│
└── proposed/
    ├── modelCNN/
    └── transformerModel/
```

---

## Environment Setup

Dependencies are specified in `Paper/requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Paper\requirements.txt
```

---

## Synthetic Dataset

To regenerate the dataset:

```powershell
cd .\Paper\dataset
python .\data.py
```

---

## Baseline Models (Raw Feature Input)

Baseline models operate directly on timestep-level inputs and typically produce per-timestep predictions.

> Note: Metrics from per-timestep models are not directly comparable to patch-level metrics unless the evaluation is performed at the same resolution.

Each baseline has its own training/evaluation scripts under:

- `Paper/baseline*/`

Run scripts from their respective directories.

---

## Proposed Pipeline: CNN Tokens + QuantFormer (Patch-Level)

### Stage A — Train CNN patch tokenizer

```powershell
cd .\Paper\proposed\modelCNN
python .\training.py
```

Output artifact:

- `trained_encoder_new.pth`

### Stage B — Export tokens

```powershell
cd .\Paper\proposed\modelCNN
python .\token_ex.py
```

Output file (consumed by the Transformer stage):

- `Paper/proposed/transformerModel/synthetic_tokens_new.pt`

It contains:

- `tokens`: `Z` with shape `(N, K, D)`
- `Y`: original timestep targets with shape `(N, T)`
- `meta`: metadata including `patch_size`

### Stage C — Train QuantFormer on tokens (patch-level forecast)

```powershell
cd .\Paper\proposed\transformerModel
python .\training.py
```

The model predicts one scalar per patch.

### Stage D — Unseen evaluation (patch-level forecast)

```powershell
cd .\Paper\proposed\transformerModel
python .\eval_unseen.py
```

---

## Metrics

For patch-level forecasting, metrics are computed over all predicted patches:

- `MSE = (1/(N·K)) · Σ_i Σ_k (y_{i,k} − ŷ_{i,k})^2`
- `MAE = (1/(N·K)) · Σ_i Σ_k |y_{i,k} − ŷ_{i,k}|`

---

## Reproducibility Notes

- The patch size `P` must be consistent across:
  - CNN training (`Paper/proposed/modelCNN/training.py`)
  - token export (`Paper/proposed/modelCNN/token_ex.py`)
  - QuantFormer training (`Paper/proposed/transformerModel/training.py` reads `meta.patch_size`)
  - evaluation (`Paper/proposed/transformerModel/eval_unseen.py`)
- Mismatched patch sizes invalidate patch-level comparisons.
- Several scripts rely on relative paths. Run them from their respective directories.

---

## Scope Note

This repository supports methodological evaluation of patch-level temporal representations. It does not implement or claim real-world trading strategies.
