# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================

import json
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd

from google.colab import drive

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/Online_Exam_Cheating")
AUDIO_DIR = PROJECT_DIR / "audio"
VISION_DIR = PROJECT_DIR / "vision_fusion_data"
RESULTS_DIR = PROJECT_DIR / "xgboost_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
STRIDE = 8

NORMAL = 0
VISION_DIM = 27
AUDIO_DIM = 2
FUSION_DIM = VISION_DIM + AUDIO_DIM

TIME_TOLERANCE = 0.05
SPLITS = ("train", "validation", "test")
RANDOM_STATE = 42
N_SPLITS = 3
N_ITER = 50

with (AUDIO_DIR / "speaker_summary.json").open(
    "r", encoding="utf-8"
) as file:
    audio_records = json.load(file)


# ============================================================
# 2. LOAD AND VALIDATE VISION ARRAYS
# ============================================================

def load_vision_split(split):
    paths = {
        "features": VISION_DIR / f"{split}_features.npy",
        "masks": VISION_DIR / f"{split}_masks.npy",
        "names": VISION_DIR / f"{split}_video_names.npy",
        "times": VISION_DIR / f"{split}_start_seconds.npy",
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    features = np.load(paths["features"], allow_pickle=False)
    masks = np.load(paths["masks"], allow_pickle=False)
    names = np.load(paths["names"], allow_pickle=True).astype(str)
    times = np.asarray(
        np.load(paths["times"], allow_pickle=False),
        dtype=np.float64,
    )

    if features.ndim != 3:
        raise ValueError(f"{split}: invalid feature shape {features.shape}")

    if features.shape[1:] != (WINDOW_SIZE, VISION_DIM):
        raise ValueError(f"{split}: unexpected feature shape {features.shape}")

    if masks.shape != features.shape[:2]:
        raise ValueError(f"{split}: mask shape mismatch")

    if len(names) != len(features) or len(times) != len(features):
        raise ValueError(f"{split}: number of windows does not match metadata")

    if not np.isfinite(features).all() or not np.isfinite(times).all():
        raise ValueError(f"{split}: NaN or infinite values detected")

    if not np.isin(masks, [0, 1]).all():
        raise ValueError(f"{split}: masks must be binary")

    return {
        "features": features.astype(np.float32, copy=False),
        "masks": masks.astype(bool, copy=False),
        "names": names,
        "times": times,
    }


data = {split: load_vision_split(split) for split in SPLITS}

for split, item in data.items():
    print(
        split,
        "| features:", item["features"].shape,
        "| masks:", item["masks"].shape,
        "| unique videos:", len(np.unique(item["names"])),
    )


# ============================================================
# 3. MANUAL FRAME LABELS
# ============================================================
# KEEP ALL ORIGINAL label_ranges_* DEFINITIONS HERE.
#
# Use the original ranges you supplied for:
#   TRAIN:      16 videos and train_ranges
#   VALIDATION:  4 videos and validation_ranges
#   TEST:       4 videos and test_ranges
#
# Do not keep the individual labels_* = [0] * length arrays.
# Do not keep the original manual range-application loops.
# The function below creates those label arrays consistently.


def make_frame_labels(length, ranges):
    """
    Convert one-based inclusive frame ranges into a zero-based
    NumPy label array.
    """
    labels = np.zeros(length, dtype=np.int32)

    for start, end, label in ranges:
        if not (1 <= start <= end <= length):
            raise ValueError(
                f"Invalid range {(start, end, label)} "
                f"for sequence length {length}"
            )

        labels[start - 1:end] = label

    return labels


# TRAIN: retain the original lengths and range lists.
train_lengths = [
    960, 879, 856, 864, 1507, 943, 920, 1114,
    1063, 860, 1143, 826, 945, 913, 1537, 1098,
]

# train_ranges must contain your 16 original range lists,
# in the exact same order as train_lengths.

# VALIDATION: retain the original range lists and lengths.
validation_lengths = [1055, 1366, 1061, 1162]

# validation_ranges must contain:
# [label_ranges_first, label_ranges_second,
#  label_ranges_third, label_ranges_forth]

# TEST: retain the original range lists and lengths.
test_lengths = [736, 928, 942, 1475]

# test_ranges must contain:
# [label_ranges_first_test, label_ranges_second_test,
#  label_ranges_third_test, label_ranges_forth_test]


def build_frame_label_sequences(lengths, ranges_list, split):
    if len(lengths) != len(ranges_list):
        raise ValueError(
            f"{split}: {len(lengths)} lengths but "
            f"{len(ranges_list)} range lists"
        )

    return [
        make_frame_labels(length, ranges)
        for length, ranges in zip(lengths, ranges_list)
    ]


# These lines require the original range lists above.
train_labels = build_frame_label_sequences(
    train_lengths, train_ranges, "train"
)

validation_labels = build_frame_label_sequences(
    validation_lengths, validation_ranges, "validation"
)

test_labels = build_frame_label_sequences(
    test_lengths, test_ranges, "test"
)

print("\nFrame-level labels created.")
for split, sequences in [
    ("train", train_labels),
    ("validation", validation_labels),
    ("test", test_labels),
]:
    print(
        split,
        "| videos:", len(sequences),
        "| lengths:", [len(x) for x in sequences],
        "| cheating frames:", sum(np.count_nonzero(x) for x in sequences),
    )


# ============================================================
# 4. BUILD TEMPORAL LABEL WINDOWS
# ============================================================

def build_temporal_labels(video_names, frame_label_sequences, split):
    video_names = np.asarray(video_names).astype(str)
    unique_videos = list(OrderedDict.fromkeys(video_names))

    if len(unique_videos) != len(frame_label_sequences):
        raise ValueError(
            f"{split}: found {len(unique_videos)} unique videos "
            f"but {len(frame_label_sequences)} label sequences. "
            "Check video order and names."
        )

    labels_by_video = dict(zip(unique_videos, frame_label_sequences))

    y = np.zeros(
        (len(video_names), WINDOW_SIZE),
        dtype=np.int32,
    )
    valid_mask = np.zeros(y.shape, dtype=bool)

    for video in unique_videos:
        indices = np.flatnonzero(video_names == video)
        frame_labels = np.asarray(
            labels_by_video[video], dtype=np.int32
        )

        # Ensure the feature windows for this video are chronological.
        video_times = data[split]["times"][indices]
        if len(video_times) > 1 and np.any(np.diff(video_times) < 0):
            raise ValueError(
                f"{split}: windows for {video} are not chronological"
            )

        for local_idx, row_idx in enumerate(indices):
            start = local_idx * STRIDE

            # This feature window starts beyond the labeled video.
            # Leave all its steps masked out instead of calling them normal.
            if start >= len(frame_labels):
                continue

            end = min(start + WINDOW_SIZE, len(frame_labels))
            window = frame_labels[start:end]
            valid_length = len(window)

            y[row_idx, :valid_length] = window
            valid_mask[row_idx, :valid_length] = True

        valid_count = sum(
            local_idx * STRIDE < len(frame_labels)
            for local_idx in range(len(indices))
        )

        if valid_count < len(indices):
            print(
                f"WARNING | {split} | {video}: "
                f"{len(indices) - valid_count} feature window(s) "
                "have no frame-label coverage and will be excluded."
            )

    print(f"\n{split.upper()} label windows")
    print("Shape:", y.shape)
    print("Valid steps:", int(valid_mask.sum()))
    print("Label values:", np.unique(y))

    return y, valid_mask


y_train, train_label_mask = build_temporal_labels(
    data["train"]["names"], train_labels, "train"
)

y_validation, validation_label_mask = build_temporal_labels(
    data["validation"]["names"], validation_labels, "validation"
)

y_test, test_label_mask = build_temporal_labels(
    data["test"]["names"], test_labels, "test"
)


# ============================================================
# 5. ALIGN AUDIO RECORDS
# ============================================================

def normalize_name(name):
    return Path(str(name)).stem.strip().casefold()


def get_value(record, possible_keys):
    for key in possible_keys:
        if key in record:
            return record[key]

    raise KeyError(f"None of these keys found: {possible_keys}")


def expand_audio_records(raw_records):
    records = []

    if isinstance(raw_records, list):
        for item in raw_records:
            records.append({
                "filename": get_value(
                    item, ["filename", "name", "video_name"]
                ),
                "start_seconds": float(get_value(
                    item, ["start_seconds", "start_time", "time"]
                )),
                "features": np.asarray(
                    item["features"], dtype=np.float32
                ),
                "mask": np.asarray(
                    get_value(item, ["mask", "masks"]),
                    dtype=np.float32,
                ),
            })

        return records

    if isinstance(raw_records, dict):
        for split in SPLITS:
            if split not in raw_records:
                continue

            part = raw_records[split]
            names = get_value(
                part, ["names", "video_names", "filenames"]
            )
            times = get_value(
                part, ["times", "start_seconds", "start_times"]
            )
            features = part["features"]
            masks = get_value(part, ["masks", "mask"])

            if not (
                len(names) == len(times)
                == len(features) == len(masks)
            ):
                raise ValueError(
                    f"Audio array lengths differ in {split}"
                )

            for name, timestamp, feat, mask in zip(
                names, times, features, masks
            ):
                records.append({
                    "filename": str(name),
                    "start_seconds": float(timestamp),
                    "features": np.asarray(feat, dtype=np.float32),
                    "mask": np.asarray(mask, dtype=np.float32),
                })

        return records

    raise TypeError(
        f"Unsupported audio data type: {type(raw_records).__name__}"
    )


audio_items = expand_audio_records(audio_records)
audio_by_name = {}

for item in audio_items:
    key = normalize_name(item["filename"])
    audio_by_name.setdefault(key, []).append(item)

aligned_audio = {}
aligned_audio_masks = {}
audio_alignment_report = {}

for split in SPLITS:
    names = data[split]["names"]
    times = data[split]["times"]

    audio_array = np.zeros(
        (len(names), WINDOW_SIZE, AUDIO_DIM),
        dtype=np.float32,
    )
    audio_mask_array = np.zeros(
        (len(names), WINDOW_SIZE),
        dtype=np.float32,
    )

    missing = []
    matched = 0

    for row, (name, timestamp) in enumerate(zip(names, times)):
        candidates = audio_by_name.get(normalize_name(name), [])

        matches = [
            item for item in candidates
            if abs(item["start_seconds"] - timestamp)
            <= TIME_TOLERANCE
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{split}: multiple audio matches for "
                f"{name} at {timestamp}"
            )

        if not matches:
            missing.append((str(name), float(timestamp)))
            continue

        item = matches[0]
        features = item["features"]
        mask = item["mask"]

        if features.shape != (WINDOW_SIZE, AUDIO_DIM):
            raise ValueError(
                f"Invalid audio shape for {name}: {features.shape}"
            )

        if mask.shape != (WINDOW_SIZE,):
            raise ValueError(
                f"Invalid audio mask shape for {name}: {mask.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite audio features for {name}")

        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Non-binary audio mask for {name}")

        audio_array[row] = features
        audio_mask_array[row] = mask
        matched += 1

    aligned_audio[split] = audio_array
    aligned_audio_masks[split] = audio_mask_array

    audio_alignment_report[split] = {
        "matched": matched,
        "missing": missing,
    }

    np.save(
        VISION_DIR / f"{split}_aligned_audio_features.npy",
        audio_array,
    )
    np.save(
        VISION_DIR / f"{split}_aligned_audio_mask.npy",
        audio_mask_array,
    )

    print(
        f"{split}: audio matched={matched}, "
        f"missing={len(missing)}"
    )


# ============================================================
# 6. BUILD FUSION ARRAYS
# ============================================================

fused_features = {}
fused_video_names = {}
fused_start_seconds = {}

for split in SPLITS:
    vision = data[split]["features"]
    audio = aligned_audio[split]

    fused = np.concatenate([vision, audio], axis=-1)

    expected_shape = (len(vision), WINDOW_SIZE, FUSION_DIM)

    if fused.shape != expected_shape:
        raise ValueError(
            f"{split}: expected {expected_shape}, got {fused.shape}"
        )

    if not np.isfinite(fused).all():
        raise ValueError(f"{split}: fused data contain NaN or infinity")

    fused_features[split] = fused
    fused_video_names[split] = data[split]["names"].copy()
    fused_start_seconds[split] = data[split]["times"].copy()

    print(f"{split} fused features:", fused.shape)


# ============================================================
# 7. PREPARE TIMESTEP-LEVEL XGBOOST DATA
# ============================================================

def prepare_xgb_split(split, labels, valid_mask):
    X = fused_features[split].reshape(-1, FUSION_DIM)

    # Both label 1 and label 2 mean cheating in the binary task.
    y = (labels.reshape(-1) != NORMAL).astype(np.int32)

    valid = valid_mask.reshape(-1)

    groups = np.repeat(
        fused_video_names[split], WINDOW_SIZE
    )

    X = X[valid]
    y = y[valid]
    groups = groups[valid]

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"{split}: X/y/group length mismatch")

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{split}: both normal and cheating samples are required"
        )

    return X, y, groups


Xtr, ytr, groups = prepare_xgb_split(
    "train", y_train, train_label_mask
)

Xv, yv, val_video_groups = prepare_xgb_split(
    "validation", y_validation, validation_label_mask
)

Xte, yte, test_video_groups = prepare_xgb_split(
    "test", y_test, test_label_mask
)

print("\nXGBoost datasets")
print("Train:", Xtr.shape, Counter(ytr.tolist()))
print("Validation:", Xv.shape, Counter(yv.tolist()))
print("Test:", Xte.shape, Counter(yte.tolist()))


# ============================================================
# 8. GROUPED CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

for fold, (train_idx, heldout_idx) in enumerate(
    cv.split(Xtr, ytr, groups=groups),
    start=1,
):
    train_videos = set(groups[train_idx])
    heldout_videos = set(groups[heldout_idx])

    if not train_videos.isdisjoint(heldout_videos):
        raise ValueError(f"Video leakage in fold {fold}")

    if len(np.unique(ytr[train_idx])) < 2:
        raise ValueError(f"Fold {fold} training split has one class")

    if len(np.unique(ytr[heldout_idx])) < 2:
        raise ValueError(f"Fold {fold} held-out split has one class")

    print(
        f"Fold {fold}: train={len(train_idx)}, "
        f"held-out={len(heldout_idx)}, "
        f"held-out videos={len(heldout_videos)}"
    )


# ============================================================
# 9. HYPERPARAMETER SEARCH
# ============================================================

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

param_space = {
    "n_estimators": [100, 200, 300, 400, 600],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 8],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10],
    "scale_pos_weight": [0.5, 1, 1.5, 2, 3, 5],
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_space,
    n_iter=N_ITER,
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True,
)

search.fit(Xtr, ytr, groups=groups)

best_model = search.best_estimator_

print("\nBest grouped CV F1:", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)


# ============================================================
# 10. OUT-OF-FOLD PROBABILITIES AND THRESHOLD SELECTION
# ============================================================

oof_probs = cross_val_predict(
    clone(best_model),
    Xtr,
    ytr,
    groups=groups,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.01):
    pred = (oof_probs >= threshold).astype(np.int32)

    threshold_results.append({
        "threshold": float(threshold),
        "Precision": precision_score(
            ytr, pred, zero_division=0
        ),
        "Recall": recall_score(
            ytr, pred, zero_division=0
        ),
        "F1": f1_score(
            ytr, pred, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)
best_row = threshold_df.loc[threshold_df["F1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\nBest OOF threshold:", round(best_threshold, 3))
print("OOF Precision:", round(best_row["Precision"], 4))
print("OOF Recall:", round(best_row["Recall"], 4))
print("OOF F1:", round(best_row["F1"], 4))

display(
    threshold_df.sort_values(
        "F1", ascending=False
    ).head(10).reset_index(drop=True).round(4)
)


# ============================================================
# 11. EVALUATION HELPER
# ============================================================

def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)

    print(f"\n{name.upper()} RESULTS")
    print("Threshold:", round(threshold, 3))
    print(
        "Precision:",
        round(precision_score(y_true, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_true, predictions, zero_division=0), 4),
    )
    print(
        "F1:",
        round(f1_score(y_true, predictions, zero_division=0), 4),
    )

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Cheating"],
            zero_division=0,
        )
    )

    if len(np.unique(y_true)) == 2:
        print("ROC-AUC:", round(roc_auc_score(y_true, probabilities), 4))
        print(
            "Average Precision:",
            round(average_precision_score(y_true, probabilities), 4),
        )

    return predictions


# ============================================================
# 12. VALIDATION EVALUATION
# ============================================================

val_probs = best_model.predict_proba(Xv)[:, 1]

val_pred = evaluate_split(
    "validation", yv, val_probs, best_threshold
)


# ============================================================
# 13. PER-VIDEO RESULTS
# ============================================================

def per_video_results(video_groups, y_true, predictions):
    results = []

    for video in np.unique(video_groups):
        idx = video_groups == video
        yt = y_true[idx]
        yp = predictions[idx]

        results.append({
            "video": video,
            "steps": int(idx.sum()),
            "normal_steps": int((yt == 0).sum()),
            "cheating_steps": int((yt == 1).sum()),
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0),
            "false_alarms": int(((yt == 0) & (yp == 1)).sum()),
            "missed_cheating": int(((yt == 1) & (yp == 0)).sum()),
        })

    return (
        pd.DataFrame(results)
        .sort_values("F1")
        .reset_index(drop=True)
    )


validation_video_results = per_video_results(
    val_video_groups, yv, val_pred
)

display(validation_video_results.round(4))


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_probs = best_model.predict_proba(Xte)[:, 1]

test_pred = evaluate_split(
    "test", yte, test_probs, best_threshold
)

test_video_results = per_video_results(
    test_video_groups, yte, test_pred
)

display(test_video_results.round(4))


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature_index": np.arange(FUSION_DIM),
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

display(importance_df.head(15).reset_index(drop=True).round(5))


# ============================================================
# 16. SAVE RESULTS
# ============================================================

threshold_df.to_csv(
    RESULTS_DIR / "training_oof_thresholds.csv",
    index=False,
)

validation_video_results.to_csv(
    RESULTS_DIR / "validation_per_video.csv",
    index=False,
)

test_video_results.to_csv(
    RESULTS_DIR / "test_per_video.csv",
    index=False,
)

importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False,
)

with (RESULTS_DIR / "model_summary.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(
        {
            "best_threshold": best_threshold,
            "best_grouped_cv_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
            "fusion_dimension": FUSION_DIM,
            "train_windows": len(fused_features["train"]),
            "validation_windows": len(fused_features["validation"]),
            "test_windows": len(fused_features["test"]),
            "audio_alignment": {
                split: {
                    "matched": audio_alignment_report[split]["matched"],
                    "missing": len(
                        audio_alignment_report[split]["missing"]
                    ),
                }
                for split in SPLITS
            },
        },
        file,
        indent=2,
    )

print("\nFinished.")
print("Fusion dimension:", FUSION_DIM)
print("Best threshold:", round(best_threshold, 3))
print("Results saved to:", RESULTS_DIR)
