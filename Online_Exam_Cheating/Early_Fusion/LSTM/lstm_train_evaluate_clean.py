"""Train and evaluate the three-step multimodal LSTM for online exam integrity.

This script uses precomputed vision/audio features and the project's annotations.
It intentionally excludes the separate XGBoost experiments and exploratory cells.
"""

import os
os.environ["PYTHONHASHSEED"] = "42"

import json
import random
from pathlib import Path
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/Online_Exam_Cheating")
AUDIO_DIR = PROJECT_DIR / "audio"
VISION_DIR = PROJECT_DIR / "vision_fusion_data"

WINDOW_SIZE = 10
STRIDE = 8
Normal = 0

VISION_DIM = 29
AUDIO_DIM = 2
FUSION_DIM = VISION_DIM + AUDIO_DIM
TIME_TOLERANCE = 0.05
SPLITS = ["train", "validation", "test"]


with open(AUDIO_DIR / "speaker_summary.json", "r") as f:
    audio_records = json.load(f)


def load_vision_split(split):
    features = np.load(
        VISION_DIR / f"{split}_features.npy",
        allow_pickle=False
    )
    masks = np.load(
        VISION_DIR / f"{split}_masks.npy",
        allow_pickle=False
    )
    names = np.load(
        VISION_DIR / f"{split}_video_names.npy",
        allow_pickle=False
    ).astype(str)

    times_path = VISION_DIR / f"{split}_start_seconds.npy"
    if not times_path.exists():
        raise FileNotFoundError(f"Missing file: {times_path}")

    times = np.asarray(
        np.load(times_path, allow_pickle=False),
        dtype=np.float64
    )

    assert features.ndim == 3
    assert features.shape[1:] == (WINDOW_SIZE, VISION_DIM)
    assert masks.shape == features.shape[:2]
    assert len(features) == len(names) == len(times)
    assert np.isfinite(features).all()
    assert np.isfinite(times).all()
    assert np.isin(masks, [0, 1]).all()

    return {
        "features": features,
        "masks": masks,
        "names": names,
