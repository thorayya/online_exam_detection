# Two-Step Multimodal LSTM — Online Exam Integrity

This script trains and evaluates the two-step LSTM experiment using precomputed vision and audio features, video-level annotations, and the fixed test threshold `0.25` used for the reported run.

## Inputs expected in Google Drive

```text
Online_Exam_Cheating/
├── audio/
│   └── speaker_summary.json
└── vision_fusion_data/
    ├── train_features.npy
    ├── train_masks.npy
    ├── train_video_names.npy
    ├── train_start_seconds.npy
    ├── validation_features.npy
    ├── validation_masks.npy
    ├── validation_video_names.npy
    ├── validation_start_seconds.npy
    ├── test_features.npy
    ├── test_masks.npy
    ├── test_video_names.npy
    └── test_start_seconds.npy
```

The annotation ranges for the 24 videos are embedded in the script. The feature arrays are expected to contain 29 vision features per timestep and the audio representation contains 2 features, producing 31 fused features per timestep. Each LSTM sample contains two consecutive windows, each flattened to 310 features.

## Outputs

The script writes the trained model, fitted scaler, and threshold to:

```text
Online_Exam_Cheating/LSTM_2steps/
├── lstm_model.keras
├── lstm_scaler.joblib
└── threshold.json
```

It also writes test predictions and per-video test metrics to the current working directory.

## Important reproducibility note

Running this script trains a new model; it does not load the previously saved model. A fixed seed improves reproducibility but does not guarantee identical results across different TensorFlow/CUDA/GPU environments. To preserve a specific trained run, keep its `.keras` model, fitted scaler, threshold file, and prediction CSVs together.

The script intentionally excludes the separate XGBoost experiments and exploratory/debugging cells. It preserves the LSTM architecture and fixed final-test threshold from the supplied code; it does not restore weights from the epoch with the highest validation AUC.
