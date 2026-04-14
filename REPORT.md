# ECG Arrhythmia Detection — Project Report

**Author:** Kevin Kirui  
**Date:** April 2026  
**Repository:** https://github.com/arapkirui513-hub/ecg-arrhythmia-detector

---

## Abstract

This project implements an end-to-end ECG arrhythmia classification pipeline using the MIT-BIH Arrhythmia Database. We compare two approaches: hand-crafted features with Random Forest, and raw waveform input to a 1D convolutional neural network (CNN). On 9,561 beats across 5 classes, the CNN achieves weighted F1 = 0.999, outperforming Random Forest (F1 = 0.996), with notable improvement on the minority atrial premature beat class (A).

---

## 1. Problem Statement

Cardiac arrhythmias affect millions worldwide. Manual ECG review is time-consuming and requires specialist expertise. Automated beat classification can assist clinicians by flagging abnormal rhythms for review, potentially reducing diagnostic delay and improving triage in resource-limited settings.

**Goal:** Build a classifier that distinguishes normal beats (N) from four arrhythmia types: left bundle branch block (L), right bundle branch block (R), premature ventricular contraction (V), and atrial premature beat (A).

---

## 2. Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet)

- 4 records selected: 100, 106, 109, 212
- Sampling rate: 360 Hz
- Lead: MLII (modified lead II)
- Total beats: 9,561 (after filtering unlabelled and rare classes)

| Class | Description | Count |
|-------|-------------|-------|
| N | Normal | 4,665 |
| L | Left bundle branch block | 2,491 |
| R | Right bundle branch block | 1,825 |
| V | Premature ventricular contraction | 547 |
| A | Atrial premature beat | 33 |

**Class imbalance:** Class A has only 33 samples, posing a challenge for classifiers.

---

## 3. Methods

### 3.1 Preprocessing

1. **Bandpass filtering:** 0.5–40 Hz Butterworth filter to remove baseline wander and high-frequency noise
2. **R-peak detection:** `gqrs_detect` from WFDB library (97.3% accuracy against ground-truth annotations)
3. **Beat segmentation:** 600ms window centered on R-peak (200ms before, 400ms after)

### 3.2 Feature Engineering (Random Forest)

8 hand-crafted features extracted per beat:

- RR interval (ms)
- RR ratio (current/previous)
- R-peak amplitude
- QRS duration
- P-wave amplitude
- T-wave amplitude
- Beat mean
- Beat standard deviation

### 3.3 Models

**Random Forest**
- 100 trees
- `class_weight='balanced'` to handle imbalance
- Train/test split: 80/20 stratified

**1D CNN**
- Input: raw beat waveform (216 samples)
- Architecture: 3 convolutional blocks (32→64→128 filters) with BatchNorm, ReLU, MaxPool
- Fully connected: 128 units + Dropout(0.5)
- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam (lr=1e-3), StepLR scheduler
- Epochs: 15

---

## 4. Results

### 4.1 Detection Accuracy

R-peak detection accuracy: **97.3%** (ground-truth annotations within ±10 samples)

### 4.2 Classification Performance

| Metric | Random Forest | 1D CNN |
|--------|---------------|--------|
| Weighted F1 | 0.996 | **0.999** |
| Class A (Atrial) F1 | 0.923 | **0.933** |
| Class L (LBBB) F1 | 0.999 | **1.000** |
| Class N (Normal) F1 | 0.998 | **0.999** |
| Class R (RBBB) F1 | 0.993 | **1.000** |
| Class V (PVC) F1 | 0.986 | **0.995** |

**Key finding:** CNN outperforms Random Forest on all classes, with the largest improvement on minority class A (+1.0% F1).

### 4.3 Confusion Matrix Analysis

The CNN confusion matrix shows:

- No false negatives on class A (atrial premature beats were all correctly classified)
- Near-perfect separation between L, R, and N classes
- Minimal confusion between V and N (3-5 samples)

This indicates the raw waveform input preserves morphological features (P-wave shape, QRS width, T-wave morphology) that hand-crafted features may miss. The CNN learned to recognise subtle P-wave differences in atrial premature beats without being explicitly programmed to extract them—a significant advantage over the Random Forest's 8 summary statistics.

---

## 5. Discussion

### 5.1 Why CNN Outperforms Random Forest

The CNN operates directly on the raw 600ms beat segment, learning filters that capture:

- P-wave morphology (subtly different in atrial premature beats)
- QRS complex width and shape (differentiates LBBB from RBBB)
- ST-segment and T-wave patterns

The Random Forest relies on 8 summary statistics, which lose temporal information. For example, two beats with identical RR intervals and R amplitudes but different QRS morphologies would appear identical to the Random Forest but distinguishable to the CNN.

### 5.2 Handling Class Imbalance

Both models used class weighting to address the severe imbalance (33 A samples vs 4,665 N samples). The CNN's superior performance on class A suggests that:

- Class weighting is more effective when combined with rich input representations
- Raw waveform provides more signal for minority class discrimination

### 5.3 Comparison to Literature

Our CNN F1 (0.999) is comparable to published results on MIT-BIH using similar architectures. However, most published work uses all 48 records and cross-validation. Our 4-record subset is insufficient for robust generalisation claims.

---

## 6. Limitations

1. **Small dataset:** Only 4 of 48 MIT-BIH records used. Results may not generalise.
2. **Single-lead ECG:** MLII only. Multi-lead ECG provides additional diagnostic information.
3. **No external validation:** Test set is from same records as training. No hold-out record evaluation.
4. **Class A sample size:** 33 samples is below recommended minimum for reliable classifier training. Results on A should be interpreted cautiously.
5. **No real-world testing:** MIT-BIH is a curated research database. Real ECGs have more noise, artefacts, and rhythm variability.

---

## 7. Conclusion & Future Work

This project demonstrates that:

- A 1D CNN on raw ECG beats outperforms classical ML on hand-crafted features
- Class weighting effectively handles imbalance
- R-peak detection at 97.3% accuracy is sufficient for beat segmentation

**Future work:**

- Expand to all 48 MIT-BIH records with cross-validation
- Add multi-lead ECG input
- Implement real-time inference pipeline
- Prospective validation on external dataset
- Explore attention mechanisms for interpretable predictions

---

## References

1. Moody, G.B. & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
2. Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 101(23), e215-e220.
3. Kiranyaz, S. et al. (2016). Real-time patient-specific ECG classification by 1-D convolutional neural networks. IEEE Transactions on Biomedical Engineering, 63(3), 653-662.

---

*Built as part of Pre-Stanmore AI for Biomedical Engineering — Week 7.*
