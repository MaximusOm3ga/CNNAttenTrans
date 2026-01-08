# Temporal Token Optimization via Attention-Based CNNs for Multivariate Time-Series Forecasting

This repository contains the reference implementation used in research on temporal tokenization for multivariate time-series learning. We study sequence-to-sequence regression on a synthetic dataset and compare against standard neural baselines as well as a Transformer (Quantformer) trained on either raw features or CNN-extracted temporal tokens.

## Task Definition

Each sample is a multivariate time series with per-timestep regression targets:

- Inputs: `X ∈ R^{N×T×F}` (samples × timesteps × features)
- Targets: `Y ∈ R^{N×T}` (per-timestep scalar regression target)

All training and evaluation scripts in this repository assume this aligned sequence-to-sequence setting.

## Code Organization

- `Transformer/dataset/`
  - `data.py`: synthetic dataset generator
  - `paper_dataset.pt`: synthetic training dataset (tensor dictionary `{X, Y}`)
  - `paper_new_test.pt`: synthetic test dataset (tensor dictionary `{X, Y}`)
- `Transformer/baselineGRU/`: GRU baseline (train/eval)
- `Transformer/baselineLSTM/`: LSTM baseline (train/eval)
- `Transformer/baselineTransModel/`: Transformer baseline (train/eval)
- `Transformer/baselineQuantTransModel/`: Quantformer baseline trained on raw features
- `Transformer/arima/`: ARIMA baseline (univariate baseline intended to operate on the target series)
- `Transformer/proposed/`
  - `modelCNN/`: DenseNet-style 1D CNN token encoder and token export
  - `transformerModel/`: Optimized Transformer trained on CNN tokens and unseen evaluation

## Environment Setup

### Dependencies

Dependencies are pinned in `Transformer/requirements.txt`.

On Windows (PowerShell):

```powershell
cd C:\Users\sauri\PycharmProjects\PythonProject1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Transformer\requirements.txt
```

Note: the requirements file pins a CUDA-enabled PyTorch build. If the specified CUDA build is incompatible with your system, install a compatible PyTorch build first, then install the remaining dependencies.

## Synthetic Dataset

Dataset files are stored under `Transformer/dataset/` as PyTorch tensor dictionaries:

```text
{"X": X, "Y": Y}
```

To (re-)generate a dataset file using the provided synthetic generator:

```powershell
cd .\Transformer\dataset
python .\data.py
```

## Baselines (Raw Features)

Many scripts use relative dataset paths (e.g., `../dataset/paper_dataset.pt`). For reproducibility, run each script from its corresponding folder, as shown below.

### GRU

Train:

```powershell
cd .\Transformer\baselineGRU
python .\training.py
```

Evaluate:

```powershell
cd .\Transformer\baselineGRU
python .\eval.py
```

### LSTM

Train:

```powershell
cd .\Transformer\baselineLSTM
python .\training.py
```

Evaluate:

```powershell
cd .\Transformer\baselineLSTM
python .\evaluation.py
```

### Transformer

Train:

```powershell
cd .\Transformer\baselineTransModel
python .\training.py
```

Evaluate:

```powershell
cd .\Transformer\baselineTransModel
python .\modelEval.py
```

### Quantformer (Raw Features)

Train:

```powershell
cd .\Transformer\baselineQuantTransModel
python .\trans_train.py
```

Evaluate:

```powershell
cd .\Transformer\baselineQuantTransModel
python .\transformer_eval.py
```

## ARIMA Baseline

The ARIMA baseline is provided under `Transformer/arima/`.

```powershell
cd .\Transformer\arima
python .\eval.py
```

## Proposed Method: CNN Temporal Tokens + Transformer

The proposed pipeline consists of:

1. Training a CNN-based token encoder (`DenseNetTokenEncoder`).
2. Exporting per-timestep token embeddings for the training dataset.
3. Training a Transformer model on token sequences.
4. Evaluating on an unseen dataset by generating tokens with the trained encoder.

### Stage A: Train the CNN token encoder

```powershell
cd .\Transformer\proposed\modelCNN
python .\training.py
```

### Stage B: Export tokens

```powershell
cd .\Transformer\proposed\modelCNN
python .\token_ex.py
```

This produces `Transformer/proposed/transformerModel/synthetic_tokens.pt`.

### Stage C: Train Transformer on tokens

```powershell
cd .\Transformer\proposed\transformerModel
python .\training.py
```

### Stage D: Evaluate on the unseen dataset

```powershell
cd .\Transformer\proposed\transformerModel
python .\eval_unseen.py
```

If `eval_unseen.py` uses an absolute path for encoder weights, update it to a valid local path (for example, `..\modelCNN\trained_encoder.pth`).

## Outputs and Metrics

Evaluation scripts report standard regression metrics, including MSE and MAE. Metrics are computed over all predicted timesteps and samples.


