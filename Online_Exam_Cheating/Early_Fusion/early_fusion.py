# ============================================================
# EARLY FUSION: VISION (27) + AUDIO (2) = 29 FEATURES
# ============================================================

import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import keras

from google.colab import drive
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/Online_Exam_Cheating")
VISION_DIR = ROOT / "vision_fusion_data"
AUDIO_JSON = ROOT / "audio" / "speaker_summary.json"

SPLITS = ("train", "validation", "test")
WINDOW_SIZE = 10
STRIDE = 8
TIME_TOLERANCE = 0.05

VISION_DIM = 27
AUDIO_DIM = 2
FUSION_DIM = VISION_DIM + AUDIO_DIM

EPOCHS = 100
BATCH_SIZE = 32

# ============================================================
# 2. FRAME-LEVEL LABEL RANGES
# Endpoints are inclusive: (start, end, label)
# ============================================================

TRAIN_LENGTHS = [
    960, 879, 856, 864, 1507, 943, 920, 1114,
    1063, 860, 1143, 826, 945, 913, 1537, 1098,
]

TRAIN_RANGES = [
    [(873, 888, 1)],
    [(829, 830, 1), (876, 877, 1)],
    [(498, 505, 1), (508, 514, 1)],
    [(498, 505, 1), (508, 514, 1)],

    [
        (1,27,1),(37,47,1),(121,139,1),(181,216,1),
        (217,226,1),(227,239,1),(244,256,1),(271,300,1),
        (346,358,1),(381,384,1),(423,437,1),(442,446,1),
        (450,461,1),(465,527,1),(530,534,1),(541,561,1),
        (627,633,1),(635,639,1),(643,669,1),(675,790,1),
        (792,832,1),(914,933,1),(996,998,1),(1000,1002,1),
        (1003,1011,1),(1013,1047,1),(1054,1060,1),
        (1063,1082,1),(1086,1090,1),(1097,1140,1),
        (1171,1192,1),(1215,1225,1),(1227,1240,1),
        (1244,1247,1),(1282,1344,1),(1349,1370,1),
        (1380,1395,1),(1397,1401,1),(1406,1506,1),
    ],

    [(937,939,1),(941,943,1)],
    [(683,684,1),(688,690,1),(693,694,1),(698,718,1)],
    [(650,670,1),(674,697,1),(1095,1099,1),(1103,1114,1)],
    [(151,164,1),(175,195,1),(1058,1059,1)],
    [(523,524,1),(528,529,1),(534,543,1),(855,856,1),(858,859,1)],

    [
        (11,15,1),(162,178,1),(183,188,1),(226,232,1),
        (320,331,1),(333,360,1),(367,380,1),(381,422,1),
        (423,450,1),(453,483,1),(488,493,1),(553,578,1),
        (657,662,1),(664,667,1),(675,685,1),(693,703,1),
        (715,752,1),(833,838,1),(853,868,1),(873,892,1),
        (897,901,1),(904,927,1),(930,950,1),(953,978,1),
        (980,981,1),(1017,1026,1),(1040,1043,1),
        (1048,1093,1),(1096,1143,1),
    ],

    [(254,255,1),(755,761,1),(765,768,1)],

    [
        (296,321,1),(399,406,1),(413,414,1),(419,420,1),
        (424,450,1),(453,454,1),(456,457,1),(460,462,1),
        (464,468,1),(471,476,1),(484,496,1),(500,503,1),
        (508,511,1),(514,517,1),(524,526,1),(529,532,1),
        (538,543,1),(545,558,1),(563,574,1),(579,594,1),
        (657,658,1),(688,690,1),(695,697,1),(700,701,1),
        (704,714,1),(722,723,1),(778,796,1),(805,806,1),
        (812,813,1),(819,827,1),(902,909,1),(912,930,1),
        (932,936,1),
    ],

    [
        (171,173,1),(257,258,1),(262,267,1),(277,282,1),
        (290,292,1),(437,439,1),(458,465,1),(548,549,1),
        (564,567,1),(576,581,1),(664,666,1),(684,685,1),
        (690,705,1),(709,730,1),
    ],

    [
        (15,18,1),(21,22,1),(24,26,1),(30,38,1),
        (194,270,1),(348,386,1),(406,412,1),(415,422,1),
        (579,584,1),(595,617,1),(632,642,1),(1047,1120,1),
        (1121,1181,1),(1183,1191,1),(1193,1224,1),
        (1227,1228,1),(1230,1250,1),(1251,1279,1),
        (1358,1372,1),(1377,1383,1),(1391,1396,1),
        (1397,1415,1),(1433,1435,1),(1441,1450,1),
        (1454,1477,1),(1480,1494,1),(1497,1504,1),
        (1512,1537,1),
    ],

    [(1,5,1),(857,860,1),(861,883,1),(1107,1108,1),(1110,1111,1)],
]

VAL_LENGTHS = [1055, 1366, 1061, 1162]

VAL_RANGES = [
    [
        (1,5,1),(8,10,1),(17,21,1),(170,179,1),
        (265,274,1),(785,821,1),(823,829,1),(841,854,1),
        (856,862,1),(875,877,1),(880,881,1),(1034,1035,1),
        (1039,1044,1),(1049,1054,1),
    ],
    [
        (2,5,1),(14,23,1),(393,395,1),(544,550,1),
        (554,566,1),(569,583,1),(595,614,1),(618,620,1),
        (627,633,1),(985,1022,1),(1033,1042,1),
        (1053,1070,1),(1080,1081,1),(1359,1362,1),
    ],
    [(476,524,2)],
    [
        (1,8,1),(14,19,1),(124,134,1),(290,292,1),
        (295,312,1),(329,333,1),(392,396,1),(499,540,1),
        (810,857,1),(980,992,1),(1139,1149,1),(1152,1161,1),
    ],
]

TEST_LENGTHS = [736, 928, 942, 1475]

TEST_RANGES = [
    [
        (1,7,1),(151,156,1),(274,288,1),(304,369,1),
        (513,568,1),(606,613,1),(674,676,1),(679,682,1),
        (685,699,1),(707,714,1),(732,735,1),
    ],
    [
        (3,4,1),(346,349,1),(358,364,1),(854,858,1),
        (860,862,1),(865,867,1),(870,871,1),(922,924,1),
    ],
    [
        (1,2,1),(3,9,1),(15,25,1),(278,286,1),(289,293,1),
        (425,433,1),(438,445,2),(449,452,1),(458,463,1),
        (466,467,1),(518,523,1),(554,556,1),(843,873,1),
        (874,888,1),(901,906,1),(937,941,1),
    ],
    [
        (1,5,1),(14,16,1),(189,212,1),(328,363,1),
        (547,555,1),(561,564,1),(569,583,1),(682,688,1),
        (697,700,1),(707,711,1),(748,831,1),(833,846,1),
        (941,1050,1),(1084,1105,1),(1109,1130,1),
        (1133,1148,1),(1151,1153,1),(1157,1160,1),
        (1164,1178,1),(1228,1235,1),(1248,1282,1),
        (1317,1326,1),(1345,1357,1),(1386,1420,1),
        (1407,1414,1),(1417,1443,1),
    ],
]

# ============================================================
# 3. LABEL AND WINDOW UTILITIES
# ============================================================

def normalize_name(name):
    """Normalize filenames consistently for matching."""
    return Path(str(name)).stem.strip().casefold()


def build_frame_labels(lengths, ranges_list):
    """Create frame labels and apply inclusive label ranges."""
    if len(lengths) != len(ranges_list):
        raise ValueError("Label lengths and range lists differ.")

    result = []

    for video_index, (length, ranges) in enumerate(
        zip(lengths, ranges_list)
    ):
        labels = np.zeros(length, dtype=np.int64)

        for start, end, label in ranges:
            if start < 0 or end < start or end >= length:
                raise ValueError(
                    f"Invalid range for video {video_index}: "
                    f"{(start, end, label)}, length={length}"
                )

            labels[start:end + 1] = label

        result.append(labels)

    return result


def binary_window_labels(window_labels, valid_masks,
                         min_positive_frames=MIN_POSITIVE_FRAMES):
    """Positive if at least N valid frames have label 1 or 2."""
    window_labels = np.asarray(window_labels)
    valid_masks = np.asarray(valid_masks, dtype=bool)

    if window_labels.shape != valid_masks.shape:
        raise ValueError("Window labels and valid masks differ in shape.")

    positive_frames = np.isin(window_labels, [1, 2]) & valid_masks
    return (positive_frames.sum(axis=1) >= min_positive_frames).astype(
        np.int32
    )


def load_vision_split(split):
    """Load and validate the original Vision arrays."""
    features = np.load(VISION_DIR / f"{split}_features.npy")
    masks = np.load(VISION_DIR / f"{split}_masks.npy")
    names = np.load(
        VISION_DIR / f"{split}_video_names.npy"
    ).astype(str)
    times = np.asarray(
        np.load(VISION_DIR / f"{split}_start_seconds.npy"),
        dtype=np.float64,
    )

    if features.ndim != 3:
        raise ValueError(f"{split}: expected 3D Vision features, got {features.shape}")

    if features.shape[1:] != (WINDOW_SIZE, VISION_DIM):
        raise ValueError(
            f"{split}: expected (N,{WINDOW_SIZE},{VISION_DIM}), "
            f"got {features.shape}"
        )

    if masks.shape != features.shape[:2]:
        raise ValueError(f"{split}: invalid Vision mask shape {masks.shape}")

    if not (len(features) == len(masks) == len(names) == len(times)):
        raise ValueError(f"{split}: feature/metadata lengths differ.")

    if not np.isfinite(features).all() or not np.isfinite(times).all():
        raise ValueError(f"{split}: NaN or Inf found in Vision data.")

    if not np.isin(masks, [0, 1]).all():
        raise ValueError(f"{split}: Vision mask must contain only 0/1.")

    return {
        "features": features.astype(np.float32),
        "masks": masks.astype(np.float32),
        "names": names,
        "times": times,
    }


# ============================================================
# 4. BUILD FRAME LABELS
# ============================================================

frame_labels_by_split = {
    "train": build_frame_labels(TRAIN_LENGTHS, TRAIN_RANGES),
    "validation": build_frame_labels(VAL_LENGTHS, VAL_RANGES),
    "test": build_frame_labels(TEST_LENGTHS, TEST_RANGES),
}

# ============================================================
# 5. LOAD AUDIO JSON: SUPPORT BOTH JSON STRUCTURES
# ============================================================

with open(AUDIO_JSON, "r", encoding="utf-8") as file:
    raw_audio = json.load(file)


def audio_dict_to_records(audio_dict):
    """Convert split-wise audio arrays into individual records."""
    records = []

    for split in SPLITS:
        if split not in audio_dict:
            continue

        block = audio_dict[split]

        # Already a list of audio records
        if isinstance(block, list):
            records.extend(block)
            continue

        if not isinstance(block, dict):
            raise ValueError(f"Unexpected audio data for split {split}.")

        # Expected split-wise structure:
        # features, masks, times, names
        required = {"features", "masks", "times", "names"}

        if required.issubset(block):
            features = block["features"]
            masks = block["masks"]
            times = block["times"]
            names = block["names"]

            if not (
                len(features) == len(masks)
                == len(times) == len(names)
            ):
                raise ValueError(
                    f"{split}: audio feature/metadata lengths differ."
                )

            for i in range(len(names)):
                records.append({
                    "filename": str(names[i]),
                    "start_seconds": float(times[i]),
                    "features": features[i],
                    "mask": masks[i],
                })
        else:
            raise ValueError(
                f"Unexpected audio JSON keys for {split}: "
                f"{list(block.keys())}"
            )

    return records


if isinstance(raw_audio, list):
    audio_records = raw_audio
elif isinstance(raw_audio, dict) and any(
    split in raw_audio for split in SPLITS
):
    audio_records = audio_dict_to_records(raw_audio)
else:
    raise ValueError(
        "Unrecognized speaker_summary.json structure."
    )


audio_by_name = {}

for record in audio_records:
    required = {"filename", "start_seconds", "features", "mask"}

    if not required.issubset(record):
        raise ValueError(
            f"Audio record missing keys: {required - set(record)}"
        )

    key = normalize_name(record["filename"])
    audio_by_name.setdefault(key, []).append(record)

print("Audio records loaded:", len(audio_records))
print("Unique audio video names:", len(audio_by_name))

# Load the Vision arrays used by the fusion step.
data = {split: load_vision_split(split) for split in SPLITS}
for split in SPLITS:
    print(f"Loaded Vision {split}: {data[split]['features'].shape}")

# ============================================================
# 6. ALIGN AUDIO AND BUILD EARLY FUSION
# ============================================================

fused_features = {}
fused_video_names = {}
fused_start_seconds = {}
fused_audio_masks = {}
fused_vision_masks = {}

for split in SPLITS:
    vision = data[split]["features"]
    names = data[split]["names"]
    times = data[split]["times"]

    aligned_audio = np.zeros(
        (len(vision), WINDOW_SIZE, AUDIO_DIM), dtype=np.float32
    )
    aligned_audio_masks = np.zeros(
        (len(vision), WINDOW_SIZE), dtype=np.float32
    )

    matched = 0
    missing = []

    for row, (video_name, start_time) in enumerate(zip(names, times)):
        candidates = audio_by_name.get(normalize_name(video_name), [])

        matches = [
            item for item in candidates
            if abs(float(item["start_seconds"]) - float(start_time))
            <= TIME_TOLERANCE
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{split}: ambiguous audio match for "
                f"{video_name} at {start_time}: {len(matches)} matches"
            )

        if not matches:
            # Keep the Vision window; represent missing audio as zeros.
            missing.append((str(video_name), float(start_time)))
            continue

        record = matches[0]
        audio_features = np.asarray(record["features"], dtype=np.float32)
        audio_mask = np.asarray(record["mask"], dtype=np.float32)

        if audio_features.shape != (WINDOW_SIZE, AUDIO_DIM):
            raise ValueError(
                f"Bad audio feature shape for {video_name}: "
                f"{audio_features.shape}"
            )

        if audio_mask.shape != (WINDOW_SIZE,):
            raise ValueError(
                f"Bad audio mask shape for {video_name}: {audio_mask.shape}"
            )

        if not np.isfinite(audio_features).all():
            raise ValueError(f"NaN/Inf in audio features for {video_name}")

        if not np.isin(audio_mask, [0, 1]).all():
            raise ValueError(f"Audio mask is not binary for {video_name}")

        aligned_audio[row] = audio_features
        aligned_audio_masks[row] = audio_mask
        matched += 1

    np.save(VISION_DIR / f"{split}_aligned_audio_features.npy", aligned_audio)
    np.save(VISION_DIR / f"{split}_aligned_audio_mask.npy", aligned_audio_masks)

    fused = np.concatenate([vision, aligned_audio], axis=-1)

    if fused.shape != (len(vision), WINDOW_SIZE, FUSION_DIM):
        raise ValueError(f"{split}: wrong fused shape: {fused.shape}")

    if not np.isfinite(fused).all():
        raise ValueError(f"{split}: NaN/Inf found in fused features")

    fused_features[split] = fused
    fused_video_names[split] = names.copy()
    fused_start_seconds[split] = times.copy()
    fused_audio_masks[split] = aligned_audio_masks
    fused_vision_masks[split] = data[split]["masks"].copy()

    print(
        f"{split}: Vision={vision.shape}, Audio={aligned_audio.shape}, "
        f"Fusion={fused.shape}, matched={matched}, missing={len(missing)}"
    )

    if missing:
        print("  First missing audio windows:", missing[:5])

# ============================================================
# 7. CHECK SPLIT METADATA AND OVERLAP
# ============================================================

for split in SPLITS:
    assert len(fused_video_names[split]) == len(fused_features[split])
    assert np.array_equal(fused_video_names[split], data[split]["names"])
    assert np.allclose(
        fused_start_seconds[split], data[split]["times"],
        atol=TIME_TOLERANCE, rtol=0,
    )

    print(
        f"{split}: {len(fused_features[split])} windows, "
        f"{len(set(fused_video_names[split]))} unique video names, "
        f"Vision valid steps={int(fused_vision_masks[split].sum())}, "
        f"Audio valid steps={int(fused_audio_masks[split].sum())}"
    )

video_sets = {
    split: {normalize_name(n) for n in data[split]["names"]}
    for split in SPLITS
}

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:
        overlap = video_sets[split_a] & video_sets[split_b]
        print(f"Video overlap {split_a}/{split_b}: {len(overlap)}")

# ============================================================
# 8. ALIGN FRAME LABELS TO WINDOW ORDER
# ============================================================

def labels_for_split(split, expected_lengths):
    """
    Associate frame-level label sequences with unique video names
    in first-seen order. Each video must have the expected number
    of windows implied by its label length and STRIDE.
    """
    names = fused_video_names[split]
    unique_names = list(dict.fromkeys(names.tolist()))
    frame_sequences = frame_labels_by_split[split]

    if len(unique_names) != len(frame_sequences):
        raise ValueError(
            f"{split}: {len(unique_names)} unique videos but "
            f"{len(frame_sequences)} frame-label sequences. "
            "Check the video ordering and label lists."
        )

    if len(expected_lengths) != len(frame_sequences):
        raise ValueError(f"{split}: label length metadata mismatch.")

    # Confirm each sequence has its declared length.
    for i, (labels, expected) in enumerate(
        zip(frame_sequences, expected_lengths)
    ):
        if len(labels) != expected:
            raise ValueError(
                f"{split}: label sequence {i} has length {len(labels)}, "
                f"expected {expected}"
            )

    label_map = {
        normalize_name(video_name): labels
        for video_name, labels in zip(unique_names, frame_sequences)
    }

    all_window_labels = []
    all_valid_masks = []

    # Preserve exact row order of the fused arrays.
    window_count_by_video = Counter(
        normalize_name(name) for name in names
    )
    positions_seen = Counter()

    for video_name in names:
        key = normalize_name(video_name)

        if key not in label_map:
            raise ValueError(f"{split}: no labels for video {video_name}")

        position = positions_seen[key]
        positions_seen[key] += 1

        frame_start = position * STRIDE
        frame_sequence = label_map[key]

        if frame_start >= len(frame_sequence):
            raise ValueError(
                f"{split}: video={video_name}, window position={position}, "
                f"start={frame_start}, label length={len(frame_sequence)}"
            )

        chunk = frame_sequence[frame_start:frame_start + WINDOW_SIZE]
        padded = np.zeros(WINDOW_SIZE, dtype=np.int64)
        valid = np.zeros(WINDOW_SIZE, dtype=bool)

        padded[:len(chunk)] = chunk
        valid[:len(chunk)] = True

        all_window_labels.append(padded)
        all_valid_masks.append(valid)

    # Validate window counts per video against label lengths.
    for name in unique_names:
        key = normalize_name(name)
        expected_windows = len(label_map[key])
        expected_windows = (expected_windows + STRIDE - 1) // STRIDE

        actual_windows = window_count_by_video[key]
        if actual_windows != expected_windows:
            raise ValueError(
                f"{split}: {name}: found {actual_windows} windows, "
                f"but label length implies {expected_windows}. "
                "Check feature-window generation and video ordering."
            )

    return (
        np.asarray(all_window_labels, dtype=np.int64),
        np.asarray(all_valid_masks, dtype=bool),
    )


windows_labels_train, masks_labels_train = labels_for_split(
    "train", TRAIN_LENGTHS
)
windows_labels_validation, masks_labels_validation = labels_for_split(
    "validation", VAL_LENGTHS
)
windows_labels_test, masks_labels_test = labels_for_split(
    "test", TEST_LENGTHS
)

y_train = binary_window_labels(
    windows_labels_train, masks_labels_train
)
y_val = binary_window_labels(
    windows_labels_validation, masks_labels_validation
)
y_test = binary_window_labels(
    windows_labels_test, masks_labels_test
)

for split, X, y in [
    ("train", fused_features["train"], y_train),
    ("validation", fused_features["validation"], y_val),
    ("test", fused_features["test"], y_test),
]:
    print(
        f"{split}: X={X.shape}, labels={y.shape}, "
        f"class counts={np.bincount(y, minlength=2)}"
    )
    if len(X) != len(y):
        raise ValueError(f"{split}: feature/label counts differ")

print("Label rule: at least", MIN_POSITIVE_FRAMES, "positive frames per window")
print("Positive labels mean original frame label 1 or 2.")

# ============================================================
# 9. EARLY FUSION AUTOENCODER
# ============================================================

class EarlyFusionAutoencoder(keras.Model):
    def __init__(self, fusion_dim=FUSION_DIM, **kwargs):
        super().__init__(**kwargs)

        self.lstm = keras.layers.LSTM(32, return_sequences=True)

        self.encoder_convs = [
            keras.layers.Conv1D(32, 3, padding="same"),
            keras.layers.Conv1D(32, 3, padding="same"),
            keras.layers.Conv1D(32, 3, padding="same"),
        ]
        self.encoder_norms = [
            keras.layers.BatchNormalization(),
            keras.layers.BatchNormalization(),
            keras.layers.BatchNormalization(),
        ]
        self.encoder_activations = [
            keras.layers.ReLU(),
            keras.layers.ReLU(),
            keras.layers.ReLU(),
        ]

        self.decoder_conv = keras.layers.Conv1DTranspose(
            32, 3, padding="same"
        )
        self.decoder_norm = keras.layers.BatchNormalization()
        self.decoder_activation = keras.layers.ReLU()

        self.output_conv = keras.layers.Conv1DTranspose(
            fusion_dim, 3, padding="same"
        )

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)

        for conv, norm, activation in zip(
            self.encoder_convs,
            self.encoder_norms,
            self.encoder_activations,
        ):
            x = conv(x)
            x = norm(x, training=training)
            x = activation(x)

        x = self.decoder_conv(x)
        x = self.decoder_norm(x, training=training)
        x = self.decoder_activation(x)

        return self.output_conv(x)


X_train = fused_features["train"]
X_val = fused_features["validation"]
X_test = fused_features["test"]

model = EarlyFusionAutoencoder(fusion_dim=FUSION_DIM)
model.build((None, WINDOW_SIZE, FUSION_DIM))

model.compile(
    optimizer="adam",
    loss=keras.losses.MeanSquaredError(),
)

model.summary()

# Preserve the original training setup: reconstruct all training windows.
model.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val, X_val),
)

# ============================================================
# 10. RECONSTRUCTION ERRORS
# ============================================================

val_reconstructed = model.predict(X_val, verbose=0)
test_reconstructed = model.predict(X_test, verbose=0)

val_errors = np.mean(
    np.square(X_val - val_reconstructed), axis=(1, 2)
)
test_errors = np.mean(
    np.square(X_test - test_reconstructed), axis=(1, 2)
)

if not np.isfinite(val_errors).all():
    raise ValueError("Validation reconstruction errors contain NaN/Inf.")

if not np.isfinite(test_errors).all():
    raise ValueError("Test reconstruction errors contain NaN/Inf.")

print("Validation reconstruction errors:", val_errors.shape)
print("Test reconstruction errors:", test_errors.shape)
print("Mean test reconstruction error:", float(test_errors.mean()))

# ============================================================
# 11. SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

if len(np.unique(y_val)) < 2:
    raise ValueError(
        f"Validation needs both classes; found {np.unique(y_val)}"
    )

threshold_candidates = np.unique(val_errors)
best_threshold = None
best_val_f1 = -1.0

for threshold in threshold_candidates:
    val_predictions = (val_errors >= threshold).astype(np.int32)
    score = f1_score(y_val, val_predictions, zero_division=0)

    if score > best_val_f1:
        best_val_f1 = score
        best_threshold = float(threshold)

# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================

y_pred = (test_errors >= best_threshold).astype(np.int32)

print("\n========== EARLY FUSION: TEST EVALUATION ==========")
print("Input shape:", X_test.shape)
print("Label rule: >= 2 valid positive frames per window")
print(f"Selected validation threshold: {best_threshold:.8f}")
print(f"Validation F1: {best_val_f1:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(
    f"Test precision: "
    f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
)
print(
    f"Test recall: "
    f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
)
print(f"Test F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")

if len(np.unique(y_test)) == 2:
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_errors):.4f}")
else:
    print("Test ROC-AUC: undefined (only one class in test labels)")

print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# ============================================================
# 13. RECONSTRUCTION ERROR DISTRIBUTION
# ============================================================

normal_errors = test_errors[y_test == 0]
cheating_errors = test_errors[y_test == 1]

plt.figure(figsize=(10, 5))

plt.hist(
    normal_errors,
    bins=30,
    alpha=0.6,
    label="Normal windows",
)

plt.hist(
    cheating_errors,
    bins=30,
    alpha=0.6,
    label="Positive windows",
)

if len(normal_errors):
    plt.axvline(
        np.mean(normal_errors),
        linestyle="--",
        label="Normal mean",
    )

if len(cheating_errors):
    plt.axvline(
        np.mean(cheating_errors),
        linestyle="--",
        label="Positive mean",
    )

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Windows")
plt.title("Early Fusion: Reconstruction Error Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n========== EARLY FUSION SUMMARY ==========")
for split in SPLITS:
    print(
        f"{split}: fusion={fused_features[split].shape}, "
        f"labels={len(y_train) if split == 'train' else len(y_val) if split == 'validation' else len(y_test)}"
    )

print("Vision dimension:", VISION_DIM)
print("Audio dimension:", AUDIO_DIM)
print("Fusion dimension:", FUSION_DIM)
print("Early Fusion pipeline completed.")
