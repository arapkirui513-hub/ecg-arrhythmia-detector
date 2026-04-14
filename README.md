# 🫀 ECG Arrhythmia Detector

> **Deep Learning for Automated Cardiac Diagnostics**

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)]()
[![WFDB](https://img.shields.io/badge/WFDB-4.1-blue)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

![Random Forest vs CNN Comparison](notebooks/figures/rf_vs_cnn_comparison.png)

---

## Quick Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 99.8% |
| Weighted F1 | **0.999** |
| R-Peak Detection | 97.3% |
| Minority Class (A) F1 | 0.933 |

---

## Clinical Significance

> **Why this matters:** Atrial Premature Beats (Class A) are early warning signs of atrial fibrillation and stroke risk. Detecting them in long-term Holter recordings manually is time-consuming. This classifier automates that screening, potentially reducing cardiologist workload by flagging high-risk segments for review.

By achieving 93.3% F1-score on the minority Atrial Premature Beat class, this system acts as a **high-sensitivity safety net** for clinicians—prioritising recall on the most clinically critical arrhythmia type.

---

## The Pipeline

```
Raw ECG (360 Hz)
        ↓
Bandpass Filter (0.5–40 Hz)
        ↓
R-Peak Detection (gqrs) → 97.3% accuracy
        ↓
Beat Segmentation (600ms window)
        ↓
┌─────────────────────┬─────────────────────┐
│   Random Forest     │       1D CNN        │
│   (8 features)      │   (raw waveform)    │
└─────────────────────┴─────────────────────┘
        ↓                       ↓
    F1: 0.996              F1: 0.999
        ↓                       ↓
        └───────────────────────┘
                    ↓
            Winner: 1D CNN
```

---

## Engineering Challenges

### 🛠 The Equilibrium Trap

**Problem:** Initial models overfitted on the 'Normal' class due to massive imbalance (4,665 vs 33 samples).

**Solution:** Implemented class weighting in CrossEntropyLoss, forcing the CNN to pay 141x more attention to Atrial Premature Beats than Normal beats.

```python
class_weights = total / (num_classes * count_per_class)
# A: 9,561 / (5 × 33) = 57.9
# N: 9,561 / (5 × 4,665) = 0.41
# Ratio: 141:1
```

This specific weighting was chosen to prioritize **high recall for Atrial Premature Beats**, ensuring that the system acts as a high-sensitivity "safety net" for clinicians—missing fewer early warning signs of atrial fibrillation.

### 🛠 Signal Noise & Baseline Wander

**Problem:** Baseline wander from patient breathing caused false R-peak detections (initial accuracy: ~88%).

**Solution:** Implemented Butterworth bandpass filter (0.5–40 Hz), removing low-frequency drift from respiration and high-frequency muscle noise.

**Result:** Detection accuracy improved from ~88% to **97.3%**.

---

## Classical vs. Deep Learning

| Aspect | Random Forest | 1D CNN | Winner |
|--------|---------------|--------|--------|
| Input | 8 hand-crafted features | Raw 600ms waveform | CNN |
| Feature Engineering | Manual (RR, QRS, amplitudes) | Learned filters | CNN |
| Minority Class (A) F1 | 0.923 | 0.933 | **CNN (+1.0%)** |
| Interpretability | Feature importance available | Requires additional tools | RF (by default) |
| Training Time | Seconds | Minutes | RF |
| Inference Time | Milliseconds | Milliseconds | Tie |

**Why CNN Won:** Raw waveform preserves temporal morphology (P-wave shape, QRS width, T-wave patterns) that 8 summary statistics lose. Two beats with identical RR intervals but different QRS morphologies are indistinguishable to RF but separable by CNN.

---

## Feature Importance (Random Forest)

*Representative rankings based on typical ECG classification literature*

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | RR_interval_ms | Highest |
| 2 | R_amplitude | High |
| 3 | QRS_duration_ms | Medium-High |
| 4 | beat_std | Medium |
| 5 | RR_ratio | Medium |
| 6 | P_wave_amplitude | Low-Medium |
| 7 | T_wave_amplitude | Low |
| 8 | beat_mean | Lowest |

---

## Results by Class

| Class | Description | RF F1 | CNN F1 | Clinical Impact |
|-------|-------------|-------|--------|-----------------|
| A | Atrial Premature Beat | 0.923 | **0.933** | Early AFib detection |
| L | Left Bundle Branch Block | 0.999 | **1.000** | Conduction abnormality |
| N | Normal | 0.998 | **0.999** | Baseline |
| R | Right Bundle Branch Block | 0.993 | **1.000** | Conduction abnormality |
| V | Premature Ventricular Contraction | 0.986 | **0.995** | Arrhythmia risk |

---

## Sensitivity vs. Specificity (Medical AI Evaluation)

| Class | Sensitivity | Specificity | Notes |
|-------|-------------|-------------|-------|
| A | 1.000 | 0.999 | No false negatives — critical for AFib screening |
| L | 1.000 | 0.999 | Excellent separation |
| N | 0.999 | 0.999 | Balanced performance |
| R | 1.000 | 0.999 | Excellent separation |
| V | 0.991 | 0.998 | Slight trade-off |

**Key insight:** High sensitivity on Class A (atrial premature beats) means we rarely miss early warning signs of atrial fibrillation—critical for stroke prevention.

*Note: Metrics calculated based on a 4-record subset of the MIT-BIH database; high values reflect the limited variability in this specific test set.*

---

## Dataset & Data Governance

**MIT-BIH Arrhythmia Database** (PhysioNet)

| Attribute | Value |
|-----------|-------|
| Source | Massachusetts Institute of Technology & Beth Israel Hospital |
| De-identification | All patient identifiers removed — approved for research use |
| Consent | Original data collected with patient consent for research purposes |
| License | Open data, citation required (Moody & Mark, 2001) |
| Usage | Educational and research purposes only |

| Class | Description | Count |
|-------|-------------|-------|
| N | Normal | 4,665 |
| L | Left Bundle Branch Block | 2,491 |
| R | Right Bundle Branch Block | 1,825 |
| V | Premature Ventricular Contraction | 547 |
| A | Atrial Premature Beat | 33 |

**Class imbalance:** Class A has only 33 samples, requiring class weighting for fair training.

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

- [REPORT.md](REPORT.md) — Full project report (2-3 pages) with methods, results, and discussion

---

## Limitations

- Only 4 of 48 MIT-BIH records used — results may not generalise
- Single-lead ECG (MLII) — multi-lead provides additional diagnostic information
- No external validation — test set from same records as training
- Class A has only 33 samples — results on A should be interpreted cautiously
- Not validated on real clinical data — MIT-BIH is a curated research database

---

## Future Work

- Expand to all 48 MIT-BIH records with cross-validation
- Multi-lead ECG input (12-lead)
- External validation on independent dataset
- Real-time inference pipeline
- Attention mechanisms for interpretable predictions
- Lung segmentation to force attention on relevant anatomy

---

## References

1. Moody, G.B. & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
2. Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215-e220.
3. Kiranyaz, S. et al. (2016). Real-time patient-specific ECG classification by 1-D convolutional neural networks. *IEEE Transactions on Biomedical Engineering*, 63(3), 653-662.

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built as part of Pre-Stanmore AI for Biomedical Engineering — Week 7.*
