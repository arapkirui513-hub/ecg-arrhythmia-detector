# ECG Arrhythmia Detector

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)]()
[![WFDB](https://img.shields.io/badge/WFDB-4.1-blue)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

> End-to-end ECG arrhythmia classification comparing Random Forest and 1D CNN on the MIT-BIH Arrhythmia Database.

**Key Result:** 1D CNN achieves **weighted F1 = 0.999**, outperforming Random Forest (F1 = 0.996), with notable improvement on minority class (Atrial Premature Beats: 0.933 vs 0.923).

---

## Quick Results

| Metric | Value |
|--------|-------|
| R-peak Detection Accuracy | 97.3% |
| Random Forest F1 | 0.996 |
| 1D CNN F1 | **0.999** |
| Total Beats | 9,561 |
| Classes | 5 (A, L, N, R, V) |

---

## Overview

This project implements a complete ECG analysis pipeline:

1. **Preprocessing:** Bandpass filter (0.5–40 Hz), R-peak detection, beat segmentation
2. **Feature Engineering:** 8 hand-crafted features for Random Forest
3. **Deep Learning:** 1D CNN on raw beat waveforms
4. **Evaluation:** Per-class F1 scores, confusion matrices, comparison

---

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet)

| Class | Description | Count |
|-------|-------------|-------|
| N | Normal | 4,665 |
| L | Left Bundle Branch Block | 2,491 |
| R | Right Bundle Branch Block | 1,825 |
| V | Premature Ventricular Contraction | 547 |
| A | Atrial Premature Beat | 33 |

---

## Installation

```bash
git clone https://github.com/arapkirui513-hub/ecg-arrhythmia-detector.git
cd ecg-arrhythmia-detector
conda create -n ai-biomed-py311 python=3.11 -y
conda activate ai-biomed-py311
pip install -r requirements.txt
```

---

## Usage

Run notebooks in order:

| Notebook | Description |
|----------|-------------|
| `01_ecg_explore.ipynb` | Load record 100, plot raw ECG |
| `02_preprocessing.ipynb` | Bandpass filter, R-peak detection, segmentation |
| `03_features.ipynb` | Feature extraction from 4 records |
| `04_classifier.ipynb` | Random Forest classifier |
| `05_cnn.ipynb` | 1D CNN classifier |

```bash
jupyter notebook notebooks/
```

---

## Results Comparison

| Class | RF F1 | CNN F1 | Δ |
|-------|-------|--------|---|
| A | 0.923 | **0.933** | +0.010 |
| L | 0.999 | **1.000** | +0.001 |
| N | 0.998 | **0.999** | +0.001 |
| R | 0.993 | **1.000** | +0.007 |
| V | 0.986 | **0.995** | +0.009 |
| **Weighted** | **0.996** | **0.999** | **+0.003** |

---

## Project Structure

```
ecg-arrhythmia-detector/
├── README.md
├── REPORT.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   ├── 01_ecg_explore.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_features.ipynb
│   ├── 04_classifier.ipynb
│   └── 05_cnn.ipynb
├── figures/
│   ├── before_after_filter.png
│   ├── filtered_with_rpeaks.png
│   ├── segmented_beats_overlay.png
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── cnn_training_curves.png
│   └── rf_vs_cnn_comparison.png
└── data/
    └── processed_record100.npz
```

---

## Documentation

- [REPORT.md](REPORT.md) — Full project report (2-3 pages)

---

## Limitations

- Only 4 of 48 MIT-BIH records used
- Single-lead ECG (MLII)
- No external validation
- Class A has only 33 samples

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built as part of Pre-Stanmore AI for Biomedical Engineering — Week 7.*
