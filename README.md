# NCD-ROM-PI — Neural Compressed-Domain ROM for Riser VIV

Data-driven reduced-order models (ROMs) for reconstructing full riser displacement fields from sparse sensor measurements in Vortex-Induced Vibration (VIV) experiments.

## Overview

This repository implements and evaluates four models for reconstructing the cross-flow (CF) and in-line (IL) displacement fields of a 38 m NDP flexible riser from a small number of point sensors:

| Model | Description |
|---|---|
| **SSM** (`LSSLStack`) | State-space sequence model with diagonal HiPPO initialisation |
| **SHRED** | Shallow Recurrent Decoder — LSTM encoder + MLP decoder |
| **POD-LS** | Instantaneous OLS linear map from sensors to POD coefficients |
| **POD-LR** | Ridge regression on the full sensor history window |

All models operate in a shared **POD latent space**: sparse sensor readings → POD modal coefficients → full displacement field via `u(x,t) ≈ ā + Φ · a(t)`.

## Repository Structure

```
NCD-ROM-PI/
├── notebooks/                  # Experiment notebooks
│   ├── VIV_unified_model_cf_pod.ipynb           # Main: QDEIM sensors, 4 models
│   ├── VIV_unified_model_cff_sensors_at_ends.ipynb  # Sensors at riser ends
│   ├── VIV_hyperparam_search_15speeds.ipynb     # Hyperparameter search
│   ├── sensor_count_study.ipynb                 # Sensor count ablation
│   ├── kdv_ssm_s4_s4d.ipynb                     # KdV equation benchmark
│   ├── beam_lssl_shred.ipynb                    # Euler–Bernoulli beam benchmark
│   └── beam_hyperparam_search.ipynb             # Beam hyperparameter study
├── scripts/
│   ├── sensing_data_augmentation.py  # Core: prepare_sensing_data() + QDEIM placement
│   ├── trajectory_model_framework.py # Per-trajectory data prep utilities
│   └── data_prep.py                  # Legacy VIV dataset builders
├── utils/
│   ├── models_s4.py        # LSSLStack (SSM), SHRED, train/predict loops
│   ├── models.py           # Baseline model definitions (POD-LS, POD-LR)
│   ├── plotting.py         # Shared plotting helpers
│   ├── animations.py       # GIF export utilities
│   └── sindy_shred.py      # Experimental SINDy-SHRED pipeline
├── NDP38m_extracted_csv/   # NDP 38 m riser CF/IL displacement CSVs
├── NDP38m_extracted/       # Pre-extracted .npz snapshots
├── model_checkpoints/      # Saved .pt checkpoints (SSM + SHRED weights + metadata)
├── figures/                # Output figures and plots
├── animations/             # Exported GIF animations
└── processed_data/         # Processed .npz datasets
```

## Dataset

Experimental measurements from the **NDP 38 m flexible riser** towing-tank tests. Each case is a riser-speed run at a velocity between 0.3 and 2.4 m/s, available for both uniform and linearly sheared current profiles.

- **CF files**: `NDP38m_extracted_csv/*_DISPLCF.csv` — cross-flow displacement `(Nx × N_t)`
- **IL files**: `NDP38m_extracted_csv/*_DISPLIL.csv` — in-line displacement `(Nx × N_t)`
- **Riser length**: 38 m, spatial resolution Nx = 47 points
- **Training set**: first 15 uniform-flow + 15 shear-flow cases; remaining cases used for extrapolation evaluation

## Sensor Placement

| Strategy | Description | Used in |
|---|---|---|
| **QDEIM** | QR-pivoted DEIM on POD basis — maximises observability | `VIV_unified_model_cf_pod.ipynb` |
| **End-of-riser** | Fixed sensors in the first/last ~20 % of riser length | `VIV_unified_model_cff_sensors_at_ends.ipynb` |

## Quick Start

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy pandas matplotlib ipywidgets scikit-learn
```

**Key package versions used:**

| Package | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.8.0 |
| NumPy | 2.1.3 |
| SciPy | 1.16.2 |
| pandas | 2.3.2 |
| matplotlib | 3.10.6 |
| scikit-learn | 1.7.2 |
| ipywidgets | 8.1.8 |

### 2. Run the main notebook

Open `notebooks/VIV_unified_model_cf_pod.ipynb` to:
- Build a QDEIM sensor dataset from all CF riser-speed cases
- Train or load pre-trained SSM and SHRED models
- Evaluate all four models across train / validation / test and extrapolation cases
- Generate RMSRE bar charts, STD profile figures, space-time contour plots, and animated GIFs

### 3. Prepare a custom dataset (CLI)

```bash
python -m scripts.sensing_data_augmentation \
    --csv NDP38m_extracted_csv/DISPL2010_DISPLCF.csv \
           NDP38m_extracted_csv/DISPL2020_DISPLCF.csv \
    --train-ratio 0.7 --valid-ratio 0.15 \
    --n-sensors 5 --n-modes 14 \
    --seq-len 100 \
    --output processed_data/my_dataset.npz
```

Key options:

| Flag | Description |
|---|---|
| `--n-sensors` | Number of QDEIM sensor locations |
| `--n-modes` | Minimum number of POD modes (exact when `--energy-threshold 0`) |
| `--energy-threshold` | Select modes to capture this % of POD energy (default 99 %) |
| `--seq-len` | Sliding window length for train split sequences |
| `--stride` | Temporal downsampling factor |
| `--no-scale-outputs` | Keep POD coefficients unscaled |

### 4. Load a checkpoint

```python
import torch
ckpt = torch.load("model_checkpoints/NDP_dataset_pod/pod_nm14_nr15_ns5_<timestamp>.pt")
# ckpt keys: model_ssm_state, model_shred_state, scaler, pod_basis, sensor_idx, ...
```

## Model Architecture

### SSM (`LSSLStack`)
- 6 LSSL layers, hidden dim = 64, state dim = 64
- Diagonal HiPPO initialisation, SiLU activation
- MLP decoder: 64 → 350 → 500 → `n_modes`

### SHRED
- 2-layer LSTM, hidden size = 64, SiLU activation
- MLP decoder: 64 → 350 → 500 → `n_modes`
- Input: sliding window of shape `(seq_len, n_sensors)`

## Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSRE** | Root Mean Squared Relative Error (%) over the full displacement field |
| **MSE** | Mean Squared Error in physical units |
| **R²** | Coefficient of determination |
| **STD RMSRE** | RMSRE of the temporal standard deviation profile (vibration envelope error) |

Metrics are computed separately for **interpolation** (within training speed range) and **extrapolation** (beyond training speed range), and for **uniform** vs. **shear** current profiles.

## Benchmarks

In addition to the NDP VIV riser:
- **KdV equation** (`kdv_ssm_s4_s4d.ipynb`) — 1-D Korteweg–de Vries travelling wave
- **Euler–Bernoulli beam** (`beam_lssl_shred.ipynb`) — forced vibration of a clamped beam

## Reference Data

High-Mode VIV reports are included under `data/` for reference:
- `2.3_High_Mode_VIV_Report_Main.pdf`
- `2.6_High_Mode_VIV_Report_Modal.pdf`
