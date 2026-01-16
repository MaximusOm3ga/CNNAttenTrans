# Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting

Reference implementation for the paper:

**Learning Patch-Level Temporal Representations for Transformer-Based Time-Series Forecasting**

This repo contains:

- A synthetic dataset generator and packaged `.pt` datasets
- Baseline models (e.g., TCN, PatchTST, GRU/LSTM/Transformer)
- The proposed two-stage pipeline:
  1) CNN patch tokenizer (autoencoder-style pretraining)
  2) Transformer (QuantFormer) forecasting on exported patch tokens

## Data / shapes

- Inputs: `X ∈ R^{N×T×F}`
- Targets: `Y ∈ R^{N×T}`

Datasets are stored as PyTorch dictionaries:

```text
{"X": X, "Y": Y}
```

## Repository structure

```text
Paper/
  dataset/
    data.py
    paper_dataset.pt
    paper_new_test.pt

  baselineTCN/
  baselinePatchTST/
  Baseline_pertimestepDL/

  proposed/
    modelCNN/
      training.py
      token_ex.py
    transformerModel/
      training.py
      eval_unseen.py
    run_pipeline.py
```

## Setup (Windows / PowerShell)

Dependencies live in `Paper/requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Paper\requirements.txt
```

## Dataset

To regenerate the synthetic dataset:

```powershell
python .\Paper\dataset\data.py
```

## Proposed pipeline (end-to-end)

The full sequence is:

1. `Paper/proposed/modelCNN/training.py` (train CNN patch tokenizer)
2. `Paper/proposed/modelCNN/token_ex.py` (export tokens to `.pt`)
3. `Paper/proposed/transformerModel/training.py` (train QuantFormer)
4. `Paper/proposed/transformerModel/eval_unseen.py` (evaluate on unseen set)

### Run everything sequentially

Use the provided orchestrator which runs each step only after the previous one finishes successfully:

```powershell
python .\Paper\proposed\run_pipeline.py
```

If any step fails (non-zero exit code), the pipeline stops and prints which step failed.

### Run steps individually

You can run each script directly:

```powershell
python .\Paper\proposed\modelCNN\training.py
python .\Paper\proposed\modelCNN\token_ex.py
python .\Paper\proposed\transformerModel\training.py
python .\Paper\proposed\transformerModel\eval_unseen.py
```

Tip: if you see import errors, run from the project root (the folder containing `README.md`) as shown above.

## Baseline models

Baselines live under `Paper/baseline*/` and `Paper/Baseline_pertimestepDL/`.

Typical usage is to `cd` into the baseline folder and run its `training.py` / `eval.py`.

## Metrics (patch-level forecasting)

Metrics are computed over all predicted patches:

- `MSE = (1/(N·K)) · Σ_i Σ_k (y_{i,k} − ŷ_{i,k})^2`
- `MAE = (1/(N·K)) · Σ_i Σ_k |y_{i,k} − ŷ_{i,k}|`

## Notes

- Patch size must be consistent across CNN training, token export, transformer training, and evaluation.
- This repo is for methodological evaluation and does not implement real-world trading strategies.
