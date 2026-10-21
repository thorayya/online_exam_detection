# Early Fusion for Online Examination Integrity

## Overview

This module implements the **Early Fusion** stage of a multimodal anomaly detection framework for online examination integrity.

The goal is to combine visual and audio features at the input level to investigate whether multimodal information can help distinguish normal examination behavior from potentially suspicious or cheating-related behavior.

## Methodology

The Early Fusion pipeline consists of the following steps:

1. **Visual features:** Load the extracted visual feature sequences.
2. **Audio features:** Load the corresponding audio feature sequences.
3. **Temporal alignment:** Align audio and visual windows using video identifiers and timestamps.
4. **Missing modality handling:** Preserve visual windows when corresponding audio features are unavailable, using zero-filled audio features and modality masks.
5. **Feature concatenation:** Concatenate visual and audio features into a unified multimodal representation.
6. **Reconstruction-based anomaly detection:** Train an autoencoder to reconstruct the fused feature sequences and use reconstruction error as an anomaly score.
7. **Evaluation:** Compare reconstruction errors for normal and cheating-related examination windows.

## Feature Configuration

| Component | Feature dimension |
|---|---:|
| Visual features | 27 |
| Audio features | 2 |
| **Fused representation** | **29** |

Each temporal window contains 10 time steps, with a stride of 8 frames.

## Experimental Results

The current experiment includes a reconstruction-error comparison between normal and cheating-related examination windows.

### Normal vs. Cheating Reconstruction Error

![Normal vs. Cheating Reconstruction Error](results/normal_vs_cheating.png)

The figure compares the reconstruction-error distributions of the two classes. It provides an initial view of whether reconstruction error can help distinguish normal examination behavior from potentially anomalous behavior.

The figure should be interpreted alongside quantitative evaluation metrics and the experimental protocol. Reconstruction error alone does not establish that a behavior is cheating.

## Repository Structure

```text
Early_Fusion/
├── early_fusion.py
├── README.md
└── results/
    └── normal_vs_cheating.png
```

## Requirements

The implementation uses Python and deep-learning and data-processing libraries, including:

- NumPy
- TensorFlow / Keras
- Matplotlib
- scikit-learn

The data-loading paths and dependencies may need to be adjusted for the local environment.

## Reproducibility

The script expects the prepared visual features, audio features, and corresponding metadata to be available at the configured paths.

The raw dataset and intermediate feature files are not included in this directory. Results depend on the dataset, preprocessing, label alignment, and model configuration used in the experiment.

## Scope and Limitations

This module represents the Early Fusion baseline. Its purpose is to establish an initial multimodal reconstruction-based anomaly detection pipeline.

Further experiments are needed to assess generalization, compare against suitable baselines, and evaluate the contribution of each modality.

## Future Work

- Quantitative evaluation using precision, recall, F1-score, and ROC-AUC.
- Comparison with vision-only and audio-only baselines.
- Investigation of modality reliability and missing-modality effects.
- Extension to intermediate fusion and temporal evidence reasoning.

---

**Project:** Multimodal Anomaly Detection for Online Examination Integrity  
**Stage:** Early Fusion
