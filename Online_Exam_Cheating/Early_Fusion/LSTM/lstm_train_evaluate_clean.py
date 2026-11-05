"""Train and evaluate an LSTM model for multimodal online exam integrity.

The script aligns precomputed audio and vision features, builds temporal
labels from the original annotations, creates video-level LSTM sequences,
and evaluates the model using a threshold selected on the validation set.
"""

import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorflow as tf
from google.colab import drive
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

PROJECT_DIR = Path("/content/drive/MyDrive/Online_Exam_Cheating")
AUDIO_DIR = PROJECT_DIR / "audio"
VISION_DIR = PROJECT_DIR / "vision_fusion_data"

WINDOW_SIZE = 10
STRIDE = 8

VISION_DIM = 29
AUDIO_DIM = 2
FUSION_DIM = VISION_DIM + AUDIO_DIM

LSTM_LENGTH = 4
N_FEATURES = 310

TIME_TOLERANCE = 0.05
SPLITS = ["train", "validation", "test"]

CHECKPOINT_PATH = PROJECT_DIR / "best_lstm_model.keras"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

tf.config.experimental.enable_op_determinism()


# ============================================================
# 3. MOUNT GOOGLE DRIVE AND LOAD DATA
# ============================================================

drive.mount("/content/drive")

with open(AUDIO_DIR / "speaker_summary.json", "r") as file:
    audio_records = json.load(file)


def load_vision_split(split):
    """Load and validate the precomputed vision features for one split."""

    features = np.load(
        VISION_DIR / f"{split}_features.npy",
        allow_pickle=False,
    )
    masks = np.load(
        VISION_DIR / f"{split}_masks.npy",
        allow_pickle=False,
    )
    names = np.load(
        VISION_DIR / f"{split}_video_names.npy",
        allow_pickle=False,
    ).astype(str)

    times_path = VISION_DIR / f"{split}_start_seconds.npy"

    if not times_path.exists():
        raise FileNotFoundError(f"Missing file: {times_path}")

    times = np.asarray(
        np.load(times_path, allow_pickle=False),
        dtype=np.float64,
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
        "times": times,
    }


data = {
    split: load_vision_split(split)
    for split in SPLITS
}


# ============================================================
# 4. ORIGINAL TRAIN ANNOTATIONS
#    Ranges are one-based and inclusive.
# ============================================================

train_lengths = [
    960, 879, 856, 864, 1507, 943, 920, 1114,
    1063, 860, 1143, 826, 945, 913, 1537, 1098
]

label_ranges_first_train = [
    (873, 888, 1),
]  # joorablo

label_ranges_second_train = [
    (829, 830, 1),
    (876, 877, 1),
]  # meowseph

label_ranges_third_train = [
    (498, 505, 1),
    (508, 514, 1),
]  # muath

label_ranges_forth_train = [
    (498, 505, 1),
    (508, 514, 1),
]  # yousef

label_ranges_fifth_train = [
    (1, 27, 1), (37, 47, 1), (121, 139, 1),
    (181, 216, 1), (217, 226, 1), (227, 239, 1),
    (244, 256, 1), (271, 300, 1), (346, 358, 1),
    (381, 384, 1), (423, 437, 1), (442, 446, 1),
    (450, 461, 1), (465, 527, 1), (530, 534, 1),
    (541, 561, 1), (627, 633, 1), (635, 639, 1),
    (643, 669, 1), (675, 790, 1), (792, 832, 1),
    (914, 933, 1), (996, 998, 1), (1000, 1002, 1),
    (1003, 1011, 1), (1013, 1047, 1), (1054, 1060, 1),
    (1063, 1082, 1), (1086, 1090, 1), (1097, 1140, 1),
    (1171, 1192, 1), (1215, 1225, 1), (1227, 1240, 1),
    (1244, 1247, 1), (1282, 1344, 1), (1349, 1370, 1),
    (1380, 1395, 1), (1397, 1401, 1), (1406, 1506, 1),
]  # yuqianyi

label_ranges_sixth_train = [
    (937, 939, 1), (941, 943, 1),
]  # zach

label_ranges_seventh_train = [
    (683, 684, 1), (688, 690, 1),
    (693, 694, 1), (698, 718, 1),
]  # xiyin

label_ranges_eighth_train = [
    (650, 670, 1), (674, 697, 1),
    (1095, 1099, 1), (1103, 1114, 1),
]  # vaibhav

label_ranges_ninth_train = [
    (151, 164, 1), (175, 195, 1), (1058, 1059, 1),
]  # vahid

label_ranges_eleventh_train = [
    (523, 524, 1), (528, 529, 1), (534, 543, 1),
    (855, 856, 1), (858, 859, 1),
]  # mustaffa

label_ranges_twelveth_train = [
    (11, 15, 1), (162, 178, 1), (183, 188, 1),
    (226, 232, 1), (320, 331, 1), (333, 360, 1),
    (367, 380, 1), (381, 422, 1), (423, 450, 1),
    (453, 483, 1), (488, 493, 1), (553, 578, 1),
    (657, 662, 1), (664, 667, 1), (675, 685, 1),
    (693, 703, 1), (715, 752, 1), (833, 838, 1),
    (853, 868, 1), (873, 892, 1), (897, 901, 1),
    (904, 927, 1), (930, 950, 1), (953, 978, 1),
    (980, 981, 1), (1017, 1026, 1), (1040, 1043, 1),
    (1048, 1093, 1), (1096, 1143, 1),
]  # wanguys

label_ranges_thirteenth_train = [
    (254, 255, 1), (755, 761, 1), (765, 768, 1),
]  # yarub

label_ranges_forteenth_train = [
    (296, 321, 1), (399, 406, 1), (413, 414, 1),
    (419, 420, 1), (424, 450, 1), (453, 454, 1),
    (456, 457, 1), (460, 462, 1), (464, 468, 1),
    (471, 476, 1), (484, 496, 1), (500, 503, 1),
    (508, 511, 1), (514, 517, 1), (524, 526, 1),
    (529, 532, 1), (538, 543, 1), (545, 558, 1),
    (563, 574, 1), (579, 594, 1), (657, 658, 1),
    (688, 690, 1), (695, 697, 1), (700, 701, 1),
    (704, 714, 1), (722, 723, 1), (778, 796, 1),
    (805, 806, 1), (812, 813, 1), (819, 827, 1),
    (902, 909, 1), (912, 930, 1), (932, 936, 1),
]  # zhoulian

label_ranges_fifteenth_train = [
    (171, 173, 1), (257, 258, 1), (262, 267, 1),
    (277, 282, 1), (290, 292, 1), (437, 439, 1),
    (458, 465, 1), (548, 549, 1), (564, 567, 1),
    (576, 581, 1), (664, 666, 1), (684, 685, 1),
    (690, 705, 1), (709, 730, 1),
]  # mark

label_ranges_sixteenth_train = [
    (15, 18, 1), (21, 22, 1), (24, 26, 1),
    (30, 38, 1), (194, 270, 1), (348, 386, 1),
    (406, 412, 1), (415, 422, 1), (579, 584, 1),
    (595, 617, 1), (632, 642, 1), (1047, 1120, 1),
    (1121, 1181, 1), (1183, 1191, 1), (1193, 1224, 1),
    (1227, 1228, 1), (1230, 1250, 1), (1251, 1279, 1),
    (1358, 1372, 1), (1377, 1383, 1), (1391, 1396, 1),
    (1397, 1415, 1), (1433, 1435, 1), (1441, 1450, 1),
    (1454, 1477, 1), (1480, 1494, 1), (1497, 1504, 1),
    (1512, 1537, 1),
]  # huangpi

label_ranges_seventeenth_train = [
    (1, 5, 1),
    (857, 860, 1),
    (861, 883, 1),
    (1087, 1088, 1),
    (1090, 1091, 1),
]  # husseinhijazi

train_ranges = [
    label_ranges_first_train,
    label_ranges_second_train,
    label_ranges_third_train,
    label_ranges_forth_train,
    label_ranges_fifth_train,
    label_ranges_sixth_train,
    label_ranges_seventh_train,
    label_ranges_eighth_train,
    label_ranges_ninth_train,
    label_ranges_eleventh_train,
    label_ranges_twelveth_train,
    label_ranges_thirteenth_train,
    label_ranges_forteenth_train,
    label_ranges_fifteenth_train,
    label_ranges_sixteenth_train,
    label_ranges_seventeenth_train,
]


# ============================================================
# 5. ORIGINAL VALIDATION ANNOTATIONS
# ============================================================

validation_lengths = [1055, 1366, 1061, 1162]

label_ranges_first = [
    (1, 5, 1), (8, 10, 1), (17, 21, 1),
    (170, 179, 1), (265, 274, 1), (785, 821, 1),
    (823, 829, 1), (841, 854, 1), (856, 862, 1),
    (875, 877, 1), (880, 881, 1), (1034, 1035, 1),
    (1039, 1044, 1), (1049, 1054, 1),
]  # alhashe

label_ranges_second = [
    (2, 5, 1), (14, 23, 1), (393, 395, 1),
    (544, 550, 1), (554, 566, 1), (569, 583, 1),
    (595, 614, 1), (618, 620, 1), (627, 633, 1),
    (985, 1022, 1), (1033, 1042, 1), (1053, 1070, 1),
    (1080, 1081, 1), (1359, 1362, 1),
]  # makiluke

label_ranges_third = [
    (476, 524, 2),
]  # chelingpi

label_ranges_forth = [
    (1, 8, 1), (14, 19, 1), (124, 134, 1),
    (290, 292, 1), (295, 312, 1), (329, 333, 1),
    (392, 396, 1), (499, 540, 1), (810, 857, 1),
    (980, 992, 1), (1139, 1149, 1), (1152, 1161, 1),
]  # bajajpak

validation_ranges = [
    label_ranges_first,
    label_ranges_second,
    label_ranges_third,
    label_ranges_forth,
]


# ============================================================
# 6. ORIGINAL TEST ANNOTATIONS
# ============================================================

test_lengths = [736, 928, 942, 1457]

label_ranges_second_test = [
    (1, 7, 1), (151, 156, 1), (274, 288, 1),
    (304, 369, 1), (513, 568, 1), (606, 613, 1),
    (674, 676, 1), (679, 682, 1), (685, 699, 1),
    (707, 714, 1), (732, 735, 1),
]  # chuangy

label_ranges_third_test = [
    (3, 4, 1), (346, 349, 1), (358, 364, 1),
    (854, 858, 1), (860, 862, 1), (865, 867, 1),
    (870, 871, 1), (922, 924, 1),
]  # cool

label_ranges_first_test = [
    (1, 2, 1), (3, 9, 1), (15, 25, 1),
    (278, 286, 1), (289, 293, 1), (425, 433, 1),
    (438, 445, 2), (449, 452, 1), (458, 463, 1),
    (466, 467, 1), (518, 523, 1), (554, 556, 1),
    (843, 873, 1), (874, 888, 1), (901, 906, 1),
    (937, 941, 1),
]  # dhingra

label_ranges_forth_test = [
    (1, 5, 1), (14, 16, 1), (189, 212, 1),
    (328, 363, 1), (547, 555, 1), (561, 564, 1),
    (569, 583, 1), (682, 688, 1), (697, 700, 1),
    (707, 711, 1), (748, 831, 1), (833, 846, 1),
    (941, 1050, 1), (1084, 1105, 1), (1109, 1130, 1),
    (1133, 1148, 1), (1151, 1153, 1), (1157, 1160, 1),
    (1164, 1178, 1), (1228, 1235, 1), (1248, 1282, 1),
    (1317, 1326, 1), (1345, 1357, 1), (1386, 1420, 1),
    (1407, 1414, 1), (1417, 1443, 1),
]  # guolingh

test_ranges = [
    label_ranges_first_test,
    label_ranges_second_test,
    label_ranges_third_test,
    label_ranges_forth_test,
]


# ============================================================
# 7. BUILD FRAME-LEVEL BINARY LABELS
# ============================================================

def build_binary_labels():
    """Create frame-level binary labels in the original split order."""

    split_lengths = {
        "train": (train_lengths, train_ranges),
        "validation": (validation_lengths, validation_ranges),
        "test": (test_lengths, test_ranges),
    }

    all_labels = []

    for split in SPLITS:
        lengths, annotation_ranges = split_lengths[split]

        if len(lengths) != len(annotation_ranges):
            raise ValueError(
                f"{split}: number of video lengths and annotations differ."
            )

        for video_idx, (length, ranges) in enumerate(
            zip(lengths, annotation_ranges)
        ):
            labels = np.zeros(length, dtype=np.int8)

            for start, end, class_id in ranges:
                assert 1 <= start <= end <= length, (
                    f"Invalid annotation in {split}, video {video_idx}: "
                    f"{(start, end, class_id)}"
                )
                labels[start - 1:end] = 1

            all_labels.append(labels)

    return all_labels


all_labels = build_binary_labels()

assert len(all_labels) == 24


# ============================================================
# 8. AUDIO RECORD NORMALIZATION
# ============================================================

def normalize_name(name):
    return Path(str(name)).stem.strip().casefold()


def get_value(record, keys):
    for key in keys:
        if key in record:
            return record[key]

    raise KeyError(f"None of the keys found: {keys}")


def expand_audio_records(raw_records):
    """Normalize supported audio-summary formats into a record list."""

    records = []

    if isinstance(raw_records, list):
        for item in raw_records:
            records.append({
                "filename": str(
                    get_value(item, ["filename", "name", "video_name"])
                ),
                "start_seconds": float(
                    get_value(item, ["start_seconds", "start_time", "time"])
                ),
                "features": np.asarray(
                    item["features"], dtype=np.float32
                ),
                "mask": np.asarray(
                    get_value(item, ["mask", "masks"]),
                    dtype=np.float32,
                ),
            })

    elif isinstance(raw_records, dict):
        if "features" in raw_records:
            split_items = [("all", raw_records)]
        else:
            split_items = [
                (split, raw_records[split])
                for split in SPLITS
                if split in raw_records
            ]

        for split, part in split_items:
            names = get_value(
                part, ["names", "video_names", "filenames"]
            )
            times = get_value(
                part, ["times", "start_seconds", "start_times"]
            )
            features = part["features"]
            masks = get_value(part, ["masks", "mask"])

            if not (
                len(names) == len(times) == len(features) == len(masks)
            ):
                raise ValueError(
                    f"Audio array lengths differ in {split}"
                )

            for name, time, feat, mask in zip(
                names, times, features, masks
            ):
                records.append({
                    "filename": str(name),
                    "start_seconds": float(time),
                    "features": np.asarray(feat, dtype=np.float32),
                    "mask": np.asarray(mask, dtype=np.float32),
                })

    else:
        raise TypeError(
            f"Unsupported audio records type: "
            f"{type(raw_records).__name__}"
        )

    return records


audio_items = expand_audio_records(audio_records)

audio_by_name = {}

for item in audio_items:
    name = normalize_name(item["filename"])
    audio_by_name.setdefault(name, []).append(item)

print("Audio records:", len(audio_items))
print("Unique audio video names:", len(audio_by_name))


# ============================================================
# 9. MAP ANNOTATIONS TO THE ORIGINAL VIDEOS
# ============================================================

expected_video_counts = {
    "train": 16,
    "validation": 4,
    "test": 4,
}

labels_by_video = {}
original_video_orders = {}

label_offset = 0

for split in SPLITS:
    names = data[split]["names"]
    video_order = list(OrderedDict.fromkeys(names))

    original_video_orders[split] = video_order

    expected_count = expected_video_counts[split]

    assert len(video_order) == expected_count, (
        f"{split}: expected {expected_count} videos, "
        f"found {len(video_order)}"
    )

    split_labels = all_labels[
        label_offset:label_offset + len(video_order)
    ]

    assert len(split_labels) == len(video_order)

    for name, labels in zip(video_order, split_labels):
        normalized_name = normalize_name(name)

        if normalized_name in labels_by_video:
            raise ValueError(
                f"Duplicate video name across splits: {name}"
            )

        labels_by_video[normalized_name] = np.asarray(
            labels,
            dtype=np.int32,
        )

        print(
            f"{split}: {name} | "
            f"annotation length={len(labels)}"
        )

    label_offset += len(video_order)

assert label_offset == len(all_labels)
assert len(labels_by_video) == 24


# ============================================================
# 10. VERIFY ORIGINAL VIDEO SPLITS
# ============================================================

video_sets = {
    split: {
        normalize_name(name)
        for name in data[split]["names"]
    }
    for split in SPLITS
}

for split in SPLITS:
    print(
        f"{split}: videos={len(video_sets[split])}, "
        f"windows={len(data[split]['features'])}, "
        f"features={data[split]['features'].shape}"
    )

assert len(video_sets["train"]) == 16
assert len(video_sets["validation"]) == 4
assert len(video_sets["test"]) == 4

assert video_sets["train"].isdisjoint(video_sets["validation"])
assert video_sets["train"].isdisjoint(video_sets["test"])
assert video_sets["validation"].isdisjoint(video_sets["test"])

assert len(
    video_sets["train"]
    | video_sets["validation"]
    | video_sets["test"]
) == 24

print("Original 16/4/4 video split verified.")


# ============================================================
# 11. ALIGN AUDIO WITH VISION
# ============================================================

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

    matched = 0
    missing = []

    for row, (name, timestamp) in enumerate(zip(names, times)):
        candidates = audio_by_name.get(normalize_name(name), [])

        matches = [
            item
            for item in candidates
            if abs(item["start_seconds"] - float(timestamp))
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
            raise ValueError(
                f"Audio features contain NaN/Inf: {name}"
            )

        if not np.isin(mask, [0, 1]).all():
            raise ValueError(
                f"Audio mask is not binary: {name}"
            )

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
    np.save(
        VISION_DIR / f"{split}_aligned_vision_features.npy",
        data[split]["features"],
    )

    print(
        f"{split}: matched={matched}, "
        f"missing={len(missing)}, "
        f"audio shape={audio_array.shape}"
    )


# ============================================================
# 12. MULTIMODAL FEATURE FUSION
# ============================================================

fused_features = {}
fused_video_names = {}
fused_start_seconds = {}
fused_audio_masks = {}

for split in SPLITS:
    vision = data[split]["features"]
    audio = aligned_audio[split]

    assert len(vision) == len(audio)

    fused = np.concatenate([vision, audio], axis=-1)

    assert fused.shape == (
        len(vision),
        WINDOW_SIZE,
        FUSION_DIM,
    )
    assert np.isfinite(fused).all()

    fused_features[split] = fused
    fused_video_names[split] = data[split]["names"].copy()
    fused_start_seconds[split] = data[split]["times"].copy()
    fused_audio_masks[split] = aligned_audio_masks[split].copy()

    print(f"{split}: fused features={fused.shape}")


# ============================================================
# 13. BUILD WINDOW-LEVEL TEMPORAL LABELS
# ============================================================

def build_split_window_labels(split):
    """Assign frame labels to each feature window."""

    names = np.asarray(
        fused_video_names[split]
    ).astype(str)

    labels = np.zeros(
        (len(names), WINDOW_SIZE),
        dtype=np.int32,
    )
    valid_mask = np.zeros(
        (len(names), WINDOW_SIZE),
        dtype=bool,
    )

    normalized_names = np.array([
        normalize_name(name)
        for name in names
    ])

    for name in OrderedDict.fromkeys(normalized_names):
        if name not in labels_by_video:
            raise KeyError(
                f"No frame labels found for video: {name}"
            )

        frame_labels = labels_by_video[name]
        row_indices = np.flatnonzero(
            normalized_names == name
        )

        for local_idx, row_idx in enumerate(row_indices):
            start = local_idx * STRIDE
            end = min(start + WINDOW_SIZE, len(frame_labels))

            if start < len(frame_labels):
                valid_length = end - start

                labels[row_idx, :valid_length] = (
                    frame_labels[start:end]
                )
                valid_mask[row_idx, :valid_length] = True

    return labels, valid_mask


window_labels = {}

for split in SPLITS:
    labels, valid_mask = build_split_window_labels(split)

    window_labels[split] = (labels, valid_mask)

    print(
        f"{split}: labels={labels.shape}, "
        f"valid positions={int(valid_mask.sum())}, "
        f"positive positions={int(labels[valid_mask].sum())}"
    )

    assert labels.shape == fused_features[split].shape[:2]
    assert valid_mask.shape == labels.shape


# ============================================================
# 14. BUILD LSTM SEQUENCES
#     Sequences never cross video boundaries.
#     A sequence is positive if any valid timestep is positive.
# ============================================================

split_times = {
    split: np.asarray(
        data[split]["times"],
        dtype=np.float64,
    )
    for split in SPLITS
}


def make_lstm_sequences(split):
    """Create fixed-length LSTM sequences within individual videos."""

    features = np.asarray(
        fused_features[split],
        dtype=np.float32,
    )
    names = np.asarray(
        fused_video_names[split]
    ).astype(str)

    labels, valid_mask = window_labels[split]

    X = []
    y = []
    video_ids = []
    time_sequences = []

    for i in range(len(features) - LSTM_LENGTH + 1):
        end = i + LSTM_LENGTH
        sequence_names = names[i:end]

        if not np.all(sequence_names == sequence_names[0]):
            continue

        sequence_features = features[i:end].reshape(
            LSTM_LENGTH,
            N_FEATURES,
        )

        sequence_labels = labels[i:end]
        sequence_valid = valid_mask[i:end]

        valid_values = sequence_labels[sequence_valid]

        if len(valid_values) == 0:
            continue

        sequence_label = int(
            np.any(valid_values == 1)
        )

        X.append(sequence_features)
        y.append(sequence_label)
        video_ids.append(sequence_names[0])
        time_sequences.append(split_times[split][i:end])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    video_ids = np.asarray(video_ids, dtype=str)
    time_sequences = np.asarray(
        time_sequences,
        dtype=np.float64,
    )

    if len(X) == 0:
        raise ValueError(
            f"No LSTM sequences created for {split}"
        )

    assert X.shape[1:] == (LSTM_LENGTH, N_FEATURES)
    assert len(X) == len(y) == len(video_ids) == len(time_sequences)
    assert np.isfinite(X).all()
    assert np.isin(y, [0, 1]).all()

    values, counts = np.unique(y, return_counts=True)

    print(f"\n{split.upper()} LSTM DATA")
    print("X shape:", X.shape)
    print(
        "Label counts:",
        dict(zip(values.tolist(), counts.tolist())),
    )
    print("Positive rate:", round(float(y.mean()), 4))
    print("Unique videos:", len(np.unique(video_ids)))

    return X, y, video_ids, time_sequences


(
    lstm_windows,
    lstm_labels,
    lstm_video_names,
    times_train,
) = make_lstm_sequences("train")

(
    lstm_windows_validation,
    lstm_labels_validation,
    lstm_video_names_validation,
    times_validation,
) = make_lstm_sequences("validation")

(
    lstm_windows_test,
    lstm_labels_test,
    lstm_video_names_test,
    times_test,
) = make_lstm_sequences("test")


if len(np.unique(lstm_labels_validation)) != 2:
    raise ValueError(
        "Validation LSTM labels must contain both classes 0 and 1."
    )

if len(np.unique(lstm_labels_test)) != 2:
    raise ValueError(
        "Test LSTM labels must contain both classes 0 and 1."
    )


# ============================================================
# 15. FEATURE SCALING
#     Fit the scaler on training data only.
# ============================================================

scaler = StandardScaler()

scaler.fit(
    lstm_windows.reshape(-1, N_FEATURES)
)


def scale_lstm_windows(X):
    """Scale each feature using statistics learned from training data."""

    scaled = scaler.transform(
        X.reshape(-1, N_FEATURES)
    ).reshape(X.shape)

    return scaled.astype(np.float32)


lstm_windows_scaled = scale_lstm_windows(lstm_windows)

lstm_windows_validation_scaled = scale_lstm_windows(
    lstm_windows_validation
)

lstm_windows_test_scaled = scale_lstm_windows(
    lstm_windows_test
)

print("\nSCALED DATA SHAPES")
print("Train:", lstm_windows_scaled.shape)
print("Validation:", lstm_windows_validation_scaled.shape)
print("Test:", lstm_windows_test_scaled.shape)


# ============================================================
# 16. BUILD THE LSTM MODEL
# ============================================================

keras.backend.clear_session()
tf.random.set_seed(SEED)

model_final = keras.Sequential([
    keras.Input(
        shape=(LSTM_LENGTH, N_FEATURES)
    ),

    layers.LSTM(
        units=32,
        dropout=0.3,
        kernel_regularizer=regularizers.l2(0.001),
    ),

    layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(0.001),
    ),
])

model_final.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

model_final.summary()


# ============================================================
# 17. TRAIN AND SAVE THE BEST MODEL
#     Model selection is based on validation AUC.
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(CHECKPOINT_PATH),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

history_final = model_final.fit(
    lstm_windows_scaled,
    lstm_labels,
    validation_data=(
        lstm_windows_validation_scaled,
        lstm_labels_validation,
    ),
    epochs=40,
    batch_size=32,
    shuffle=True,
    callbacks=[checkpoint],
    verbose=1,
)

best_epoch = int(
    np.argmax(history_final.history["val_auc"]) + 1
)

best_history_auc = float(
    np.max(history_final.history["val_auc"])
)

model_final = keras.models.load_model(
    CHECKPOINT_PATH
)

print("\nBest epoch by validation AUC:", best_epoch)
print("Best validation AUC during training:", best_history_auc)


# ============================================================
# 18. VALIDATION AND THRESHOLD SELECTION
# ============================================================

val_probs = model_final.predict(
    lstm_windows_validation_scaled,
    verbose=0,
).ravel()

validation_auc = roc_auc_score(
    lstm_labels_validation,
    val_probs,
)

threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.05):
    val_preds = (
        val_probs >= threshold
    ).astype(np.int8)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            lstm_labels_validation,
            val_preds,
            zero_division=0,
        ),
        "recall": recall_score(
            lstm_labels_validation,
            val_preds,
            zero_division=0,
        ),
        "f1": f1_score(
            lstm_labels_validation,
            val_preds,
            zero_division=0,
        ),
    })

threshold_results.sort(
    key=lambda row: row["f1"],
    reverse=True,
)

print("\nValidation AUC:", validation_auc)
print("\nTop validation thresholds:")

for row in threshold_results[:10]:
    print(row)

best_threshold = threshold_results[0]["threshold"]

print("\nSelected validation threshold:", best_threshold)

val_preds = (
    val_probs >= best_threshold
).astype(np.int8)

print("\nValidation confusion matrix:")
print(
    confusion_matrix(
        lstm_labels_validation,
        val_preds,
        labels=[0, 1],
    )
)

print("\nValidation classification report:")
print(
    classification_report(
        lstm_labels_validation,
        val_preds,
        labels=[0, 1],
        target_names=["Normal", "Cheating"],
        zero_division=0,
    )
)


# ============================================================
# 19. FINAL TEST EVALUATION
#     Use the threshold selected on the validation set.
# ============================================================

test_probs = model_final.predict(
    lstm_windows_test_scaled,
    verbose=0,
).ravel()

test_preds = (
    test_probs >= best_threshold
).astype(np.int8)

test_auc = roc_auc_score(
    lstm_labels_test,
    test_probs,
)

test_precision = precision_score(
    lstm_labels_test,
    test_preds,
    zero_division=0,
)

test_recall = recall_score(
    lstm_labels_test,
    test_preds,
    zero_division=0,
)

test_f1 = f1_score(
    lstm_labels_test,
    test_preds,
    zero_division=0,
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print("Selected threshold:", best_threshold)
print(f"Test AUC:       {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")

print("\nTest confusion matrix:")
print(
    confusion_matrix(
        lstm_labels_test,
        test_preds,
        labels=[0, 1],
    )
)

print("\nTest classification report:")
print(
    classification_report(
        lstm_labels_test,
        test_preds,
        labels=[0, 1],
        target_names=["Normal", "Cheating"],
        zero_division=0,
    )
)
