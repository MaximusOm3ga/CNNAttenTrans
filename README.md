# Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting

This repository provides the reference implementation accompanying the paper  
**“Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting.”**

The codebase includes several deep learning baselines, a classical ARIMA reference model, and a proposed **two-stage patch-based forecasting pipeline** that combines convolutional tokenization with Transformer-based sequence modelling.

---

## Problem Setup

We consider a supervised regression task on multivariate time-series data.

### Data representation

- **Inputs**:

  $$\mathbf{X} \in \mathbb{R}^{N \times T \times F}$$

  where:

  - $N$ is the number of samples,
  - $T$ is the sequence length (timesteps),
  - $F$ is the number of input features.

- **Targets**:

  $$\mathbf{Y} \in \mathbb{R}^{N \times T}$$

  where each timestep has a scalar regression target.

---

## Patch-Level Learning (Proposed Method)

The proposed method operates on **temporal patches** rather than individual timesteps.

Given a patch size $P$:

- Each sequence is segmented into

  $$K = \left\lfloor \frac{T}{P} \right\rfloor$$

  non-overlapping temporal patches.

### Patch encoding

A convolutional neural network (CNN) encodes each patch into a fixed-dimensional token embedding:

- **Patch-level tokens**:

  $$\mathbf{Z} \in \mathbb{R}^{N \times K \times D}$$

Token-level self-attention is applied **after** convolutional encoding to refine patch representations before Transformer processing.

### Patch-level targets

For fair training and evaluation at the same temporal resolution, timestep-level targets are aggregated within each patch:

- **Patch-level targets**:

  $$\mathbf{Y}_{\text{patch}} \in \mathbb{R}^{N \times K}$$

Aggregation is performed consistently (e.g., patch mean or last timestep).

> **Important**  
> The proposed method predicts **one value per patch (token)**, not per timestep.

---

## Repository Structure

```text
Transformer/
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

Dependencies are specified in `Transformer/requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Transformer\requirements.txt
```

---

## Synthetic Dataset

Datasets are stored as PyTorch dictionaries:

```text
{"X": X, "Y": Y}
```

To regenerate the dataset:

```powershell
cd .\Transformer\dataset
python .\data.py
```

---

## Baseline Models (Raw Feature Input)

Baseline models operate directly on timestep-level inputs and produce per-timestep predictions.

> **Note**  
> Metrics from these models are not directly comparable to patch-level metrics unless evaluated at the same resolution.

### GRU / LSTM / Transformer / QuantFormer (raw features)

Each baseline has its own training and evaluation scripts under:

- `Transformer/baseline*/`

Run scripts from their respective directories.

---

## ARIMA Baseline

The ARIMA model is provided as a classical statistical reference.

- Operates in a univariate setting using only the target series.
- Uses rolling one-step-ahead forecasting.
- Included for contextual comparison only.

```powershell
cd .\Transformer\arima
python .\eval.py
```

---

## Proposed Pipeline: CNN Tokens + Transformer (Patch-Level)

### Stage A — Train CNN Patch Tokenizer

```powershell
cd .\Transformer\proposed\modelCNN
python .\training.py
```

Artifact:

- `trained_encoder_new.pth`

### Stage B — Export Tokens

```powershell
cd .\Transformer\proposed\modelCNN
python .\token_ex.py
```

Output file contains:

- `tokens`: $(N, K, D)$
- `Y`: original timestep-level targets $(N, T)$
- `meta`: metadata (including `patch_size`)

### Stage C — Train Transformer on Tokens

```powershell
cd .\Transformer\proposed\transformerModel
python .\training.py
```

The model predicts one scalar per patch.

### Stage D — Unseen Evaluation (Patch-Level)

```powershell
cd .\Transformer\proposed\transformerModel
python .\eval_unseen.py
```

---

## Metrics

For the proposed patch-level pipeline, metrics are computed over all patches:

$$\mathrm{MSE}=\frac{1}{N K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left(y_{i,k} - \hat{y}_{i,k}\right)^2$$

$$\mathrm{MAE}=\frac{1}{N K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left|y_{i,k} - \hat{y}_{i,k}\right|$$

---

## Reproducibility Notes

- The patch size $P$ must be identical across:
  - CNN training
  - token export
  - Transformer training
  - evaluation
- Mismatched patch sizes invalidate results.
- Many scripts rely on relative paths; run them from their respective directories.

---

## Scope Note

This repository supports methodological evaluation of patch-level temporal representations. It does not implement or claim real-world trading strategies.
