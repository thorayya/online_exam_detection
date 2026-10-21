# ============================================================
# 1. IMPORTS, DRIVE AND CONFIGURATION
# ============================================================

import json
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/Online_Exam_Cheating")
AUDIO_DIR = PROJECT_DIR / "audio"
VISION_DIR = PROJECT_DIR / "vision_fusion_data"
RESULTS_DIR = PROJECT_DIR / "xgboost_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
STRIDE = 8
Normal = 0

VISION_DIM = 27
AUDIO_DIM = 2
FUSION_DIM = VISION_DIM + AUDIO_DIM
TIME_TOLERANCE = 0.05
SPLITS = ["train", "validation", "test"]

with open(AUDIO_DIR / "speaker_summary.json", "r") as f:
    audio_records = json.load(f)


# ============================================================
# 2. LOAD VISION ARRAYS
# ============================================================

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
        "times": times,
    }


data = {s: load_vision_split(s) for s in SPLITS}

for split in SPLITS:
    print(
        split,
        "| features:", data[split]["features"].shape,
        "| masks:", data[split]["masks"].shape,
        "| unique videos:",
        len(set(data[split]["names"]))
    )


# ============================================================
# 3. ORIGINAL TRAIN ANNOTATIONS
# Ranges are one-based and inclusive.
# ============================================================

train_lengths = [
    960, 879, 856, 864, 1507, 943, 920, 1114,
    1063, 860, 1143, 826, 945, 913, 1537, 1098
]

label_ranges_first_train = [
    (873, 888, 1),
]

label_ranges_second_train = [
    (829, 830, 1),
    (876, 877, 1),
]

label_ranges_third_train = [
    (498, 505, 1),
    (508, 514, 1),
]

label_ranges_forth_train = [
    (498, 505, 1),
    (508, 514, 1),
]

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
]

label_ranges_sixth_train = [
    (937, 939, 1), (941, 943, 1),
]

label_ranges_seventh_train = [
    (683, 684, 1), (688, 690, 1),
    (693, 694, 1), (698, 718, 1),
]

label_ranges_eighth_train = [
    (650, 670, 1), (674, 697, 1),
    (1095, 1099, 1), (1103, 1114, 1),
]

label_ranges_ninth_train = [
    (151, 164, 1), (175, 195, 1), (1058, 1059, 1),
]

label_ranges_eleventh_train = [
    (523, 524, 1), (528, 529, 1), (534, 543, 1),
    (855, 856, 1), (858, 859, 1),
]

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
]

label_ranges_thirteenth_train = [
    (254, 255, 1), (755, 761, 1), (765, 768, 1),
]

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
]

label_ranges_fifteenth_train = [
    (171, 173, 1), (257, 258, 1), (262, 267, 1),
    (277, 282, 1), (290, 292, 1), (437, 439, 1),
    (458, 465, 1), (548, 549, 1), (564, 567, 1),
    (576, 581, 1), (664, 666, 1), (684, 685, 1),
    (690, 705, 1), (709, 730, 1),
]

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
]

label_ranges_seventeenth_train = [
    (1, 5, 1), (857, 860, 1), (861, 883, 1),
    (1107, 1108, 1), (1110, 1111, 1),
]

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
# 4. ORIGINAL VALIDATION ANNOTATIONS
# ============================================================

validation_lengths = [1055, 1366, 1061, 1162]

label_ranges_first = [
    (1, 5, 1), (8, 10, 1), (17, 21, 1),
    (170, 179, 1), (265, 274, 1), (785, 821, 1),
    (823, 829, 1), (841, 854, 1), (856, 862, 1),
    (875, 877, 1), (880, 881, 1), (1034, 1035, 1),
    (1039, 1044, 1), (1049, 1054, 1),
]

label_ranges_second = [
    (2, 5, 1), (14, 23, 1), (393, 395, 1),
    (544, 550, 1), (554, 566, 1), (569, 583, 1),
    (595, 614, 1), (618, 620, 1), (627, 633, 1),
    (985, 1022, 1), (1033, 1042, 1), (1053, 1070, 1),
    (1080, 1081, 1), (1359, 1362, 1),
]

label_ranges_third = [
    (476, 524, 2),
]

label_ranges_forth = [
    (1, 8, 1), (14, 19, 1), (124, 134, 1),
    (290, 292, 1), (295, 312, 1), (329, 333, 1),
    (392, 396, 1), (499, 540, 1), (810, 857, 1),
    (980, 992, 1), (1139, 1149, 1), (1152, 1161, 1),
]

validation_ranges = [
    label_ranges_first,
    label_ranges_second,
    label_ranges_third,
    label_ranges_forth,
]


# ============================================================
# 5. ORIGINAL TEST ANNOTATIONS
# ============================================================

test_lengths = [736, 928, 942, 1475]

label_ranges_first_test = [
    (1, 7, 1), (151, 156, 1), (274, 288, 1),
    (304, 369, 1), (513, 568, 1), (606, 613, 1),
    (674, 676, 1), (679, 682, 1), (685, 699, 1),
    (707, 714, 1), (732, 735, 1),
]

label_ranges_second_test = [
    (3, 4, 1), (346, 349, 1), (358, 364, 1),
    (854, 858, 1), (860, 862, 1), (865, 867, 1),
    (870, 871, 1), (922, 924, 1),
]

label_ranges_third_test = [
    (1, 2, 1), (3, 9, 1), (15, 25, 1),
    (278, 286, 1), (289, 293, 1), (425, 433, 1),
    (438, 445, 2), (449, 452, 1), (458, 463, 1),
    (466, 467, 1), (518, 523, 1), (554, 556, 1),
    (843, 873, 1), (874, 888, 1), (901, 906, 1),
    (937, 941, 1),
]

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
]

test_ranges = [
    label_ranges_first_test,
    label_ranges_second_test,
    label_ranges_third_test,
    label_ranges_forth_test,
]


# ============================================================
# 6. BUILD FRAME LABELS SAFELY
# Keep every original range; clip only at sequence boundaries.
# ============================================================

def make_frame_labels(length, ranges, video_name, split):
    labels = np.zeros(length, dtype=np.int32)

    for start, end, label in ranges:
        if start < 1 or end < start:
            raise ValueError(
                f"{split}/{video_name}: malformed range "
                f"{(start, end, label)}"
            )

        if end > length:
            print(
                f"WARNING | {split} | {video_name}: "
                f"range {(start, end, label)} exceeds length {length}; "
                f"only frames inside the sequence are labeled."
            )

        clipped_start = max(start, 1)
        clipped_end = min(end, length)

        if clipped_start <= clipped_end:
            labels[clipped_start - 1:clipped_end] = label

    return labels


def build_label_sequences(lengths, ranges_list, split):
    if len(lengths) != len(ranges_list):
        raise ValueError(
            f"{split}: {len(lengths)} lengths but "
            f"{len(ranges_list)} annotation lists."
        )

    sequences = []
    for i, (length, ranges) in enumerate(zip(lengths, ranges_list)):
        sequences.append(
            make_frame_labels(
                length, ranges, f"annotation_{i + 1}", split
            )
        )

    return sequences


train_labels = build_label_sequences(
    train_lengths, train_ranges, "train"
)
validation_labels = build_label_sequences(
    validation_lengths, validation_ranges, "validation"
)
test_labels = build_label_sequences(
    test_lengths, test_ranges, "test"
)

for split, seqs in [
    ("train", train_labels),
    ("validation", validation_labels),
    ("test", test_labels),
]:
    print(
        f"{split}: videos={len(seqs)}, "
        f"lengths={[len(x) for x in seqs]}, "
        f"positive frames={sum(np.count_nonzero(x) for x in seqs)}"
    )


# ============================================================
# 7. VALIDATE VIDEO ORDER AND BUILD WINDOW LABELS
# ============================================================

def build_temporal_labels(video_names, frame_label_sequences, split):
    video_names = np.asarray(video_names).astype(str)
    unique_videos = list(OrderedDict.fromkeys(video_names))

    if len(unique_videos) != len(frame_label_sequences):
        raise ValueError(
            f"{split}: found {len(unique_videos)} unique feature videos "
            f"but {len(frame_label_sequences)} label sequences.\n"
            f"Feature video order: {unique_videos}"
        )

    y = np.zeros(
        (len(video_names), WINDOW_SIZE),
        dtype=np.int32
    )
    label_mask = np.zeros_like(y, dtype=bool)

    cursor = 0
    for video, frame_labels in zip(unique_videos, frame_label_sequences):
        indices = np.flatnonzero(video_names == video)
        frame_labels = np.asarray(frame_labels, dtype=np.int32)

        # Check whether this video's feature windows are contiguous.
        if len(indices) == 0:
            raise ValueError(f"{split}: no windows found for {video}")

        # The first window starts at the beginning of its sequence.
        # Each subsequent window advances by STRIDE.
        for local_idx, row_idx in enumerate(indices):
            start = local_idx * STRIDE

            if start >= len(frame_labels):
                print(
                    f"WARNING | {split} | {video}: window {local_idx} "
                    f"starts at frame {start}, beyond label length "
                    f"{len(frame_labels)}; label mask remains invalid."
                )
                continue

            end = min(start + WINDOW_SIZE, len(frame_labels))
            window = frame_labels[start:end]
            valid_len = len(window)

            y[row_idx, :valid_len] = window
            label_mask[row_idx, :valid_len] = True

        expected_windows = (len(frame_labels) + STRIDE - 1) // STRIDE
        if len(indices) != expected_windows:
            print(
                f"WARNING | {split} | {video}: feature windows={len(indices)}, "
                f"expected approximately {expected_windows} from label length "
                f"{len(frame_labels)}. Check window generation and video order."
            )

        cursor += len(indices)

    print(f"\n{split.upper()} LABEL AUDIT")
    print("Unique videos:", len(unique_videos))
    print("Windows:", len(video_names))
    print("Label shape:", y.shape)
    print("Valid label steps:", int(label_mask.sum()))
    print("Positive labeled steps:", int(((y == 1) & label_mask).sum()))
    print("Classes among valid steps:", np.unique(y[label_mask]))

    return y, label_mask


train_video_names = data["train"]["names"]
validation_video_names = data["validation"]["names"]
test_video_names = data["test"]["names"]

train_features = data["train"]["features"]
validation_features = data["validation"]["features"]
test_features = data["test"]["features"]

y_train, train_label_mask = build_temporal_labels(
    train_video_names, train_labels, "train"
)
y_validation, validation_label_mask = build_temporal_labels(
    validation_video_names, validation_labels, "validation"
)
y_test, test_label_mask = build_temporal_labels(
    test_video_names, test_labels, "test"
)

assert y_train.shape == (len(train_features), WINDOW_SIZE)
assert y_validation.shape == (len(validation_features), WINDOW_SIZE)
assert y_test.shape == (len(test_features), WINDOW_SIZE)


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
                "features": np.asarray(item["features"], dtype=np.float32),
                "mask": np.asarray(
                    get_value(item, ["mask", "masks"]), dtype=np.float32
                ),
            })
        return records

    if isinstance(raw_records, dict):
        # Handle either a split-wise structure or a single record.
        if "features" in raw_records:
            split_items = [("unknown", raw_records)]
        else:
            split_items = [
                (s, raw_records[s]) for s in SPLITS if s in raw_records
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

            if not (len(names) == len(times) == len(features) == len(masks)):
                raise ValueError(f"Audio array lengths differ in {split}")

            for name, time, feat, mask in zip(names, times, features, masks):
                records.append({
                    "filename": str(name),
                    "start_seconds": float(time),
                    "features": np.asarray(feat, dtype=np.float32),
                    "mask": np.asarray(mask, dtype=np.float32),
                })
        return records

    raise TypeError(
        f"Unsupported audio records type: {type(raw_records).__name__}"
    )


audio_items = expand_audio_records(audio_records)
audio_by_name = {}

for item in audio_items:
    audio_by_name.setdefault(
        normalize_name(item["filename"]), []
    ).append(item)

print("\nAudio records:", len(audio_items))
print("Unique normalized names:", len(audio_by_name))


# ============================================================
# 9. ALIGN AUDIO WITH VISION
# ============================================================

aligned_audio = {}
aligned_audio_masks = {}
audio_alignment_report = {}

for split in SPLITS:
    names = data[split]["names"]
    times = data[split]["times"]

    audio_array = np.zeros(
        (len(names), WINDOW_SIZE, AUDIO_DIM), dtype=np.float32
    )
    audio_mask_array = np.zeros(
        (len(names), WINDOW_SIZE), dtype=np.float32
    )

    matched = 0
    missing = []

    for row, (name, timestamp) in enumerate(zip(names, times)):
        candidates = audio_by_name.get(normalize_name(name), [])

        matches = [
            item for item in candidates
            if abs(item["start_seconds"] - float(timestamp))
            <= TIME_TOLERANCE
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{split}: multiple audio matches for {name} "
                f"at {timestamp}: {len(matches)}"
            )

        if not matches:
            missing.append((str(name), float(timestamp)))
            continue

        item = matches[0]
        feat = item["features"]
        mask = item["mask"]

        if feat.shape != (WINDOW_SIZE, AUDIO_DIM):
            raise ValueError(
                f"Invalid audio shape for {name}: {feat.shape}"
            )
        if mask.shape != (WINDOW_SIZE,):
            raise ValueError(
                f"Invalid audio mask shape for {name}: {mask.shape}"
            )
        if not np.isfinite(feat).all():
            raise ValueError(f"Audio features contain NaN/Inf: {name}")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Audio mask is not binary: {name}")

        audio_array[row] = feat
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
        audio_array
    )
    np.save(
        VISION_DIR / f"{split}_aligned_audio_mask.npy",
        audio_mask_array
    )

    print(
        f"{split} audio: matched={matched}, missing={len(missing)}, "
        f"shape={audio_array.shape}"
    )


# ============================================================
# 10. MULTIMODAL FUSION
# ============================================================

fused_features = {}
fused_video_names = {}
fused_start_seconds = {}
fused_vision_masks = {}
fused_audio_masks = {}

for split in SPLITS:
    vision = data[split]["features"]
    audio = aligned_audio[split]
    fused = np.concatenate([vision, audio], axis=-1)

    assert fused.shape == (len(vision), WINDOW_SIZE, FUSION_DIM)
    assert np.isfinite(fused).all()

    fused_features[split] = fused
    fused_video_names[split] = data[split]["names"].copy()
    fused_start_seconds[split] = data[split]["times"].copy()
    fused_vision_masks[split] = data[split]["masks"].copy()
    fused_audio_masks[split] = aligned_audio_masks[split].copy()

    print(split, "fusion shape:", fused.shape)


# ============================================================
# 11. PREPARE XGBOOST DATA
# ============================================================

X_train = fused_features["train"]
X_val = fused_features["validation"]
X_test = fused_features["test"]

X_train_flat = X_train.reshape(-1, FUSION_DIM)
X_val_flat = X_val.reshape(-1, FUSION_DIM)
X_test_flat = X_test.reshape(-1, FUSION_DIM)

# Labels 1 and 2 are both treated as the positive class.
y_train_flat = (y_train.reshape(-1) != Normal).astype(np.int32)
y_val_flat = (y_validation.reshape(-1) != Normal).astype(np.int32)
y_test_flat = (y_test.reshape(-1) != Normal).astype(np.int32)

train_valid = train_label_mask.reshape(-1)
val_valid = validation_label_mask.reshape(-1)
test_valid = test_label_mask.reshape(-1)

Xtr = X_train_flat[train_valid]
ytr = y_train_flat[train_valid]

Xv = X_val_flat[val_valid]
yv = y_val_flat[val_valid]

Xte = X_test_flat[test_valid]
yte = y_test_flat[test_valid]

groups_train_all = np.repeat(
    fused_video_names["train"], WINDOW_SIZE
)
groups_val_all = np.repeat(
    fused_video_names["validation"], WINDOW_SIZE
)
groups_test_all = np.repeat(
    fused_video_names["test"], WINDOW_SIZE
)

groups = groups_train_all[train_valid]
val_video_groups = groups_val_all[val_valid]
test_video_groups = groups_test_all[test_valid]

assert len(Xtr) == len(ytr) == len(groups)
assert len(Xv) == len(yv) == len(val_video_groups)
assert len(Xte) == len(yte) == len(test_video_groups)

print("\nXGBoost data")
print("Train:", Xtr.shape, Counter(ytr.tolist()))
print("Validation:", Xv.shape, Counter(yv.tolist()))
print("Test:", Xte.shape, Counter(yte.tolist()))
print("Training videos:", len(np.unique(groups)))


# ============================================================
# 12. MODELING IMPORTS
# ============================================================

from xgboost import XGBClassifier

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

from sklearn.base import clone


# ============================================================
# 13. GROUPED CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

for fold, (tr_idx, va_idx) in enumerate(
    cv.split(Xtr, ytr, groups=groups), start=1
):
    train_videos = set(groups[tr_idx])
    heldout_videos = set(groups[va_idx])

    assert train_videos.isdisjoint(heldout_videos)

    print(
        f"Fold {fold}: train={len(tr_idx)}, "
        f"held-out={len(va_idx)}, "
        f"held-out videos={len(heldout_videos)}, "
        f"train classes={np.unique(ytr[tr_idx])}, "
        f"held-out classes={np.unique(ytr[va_idx])}"
    )

    if len(np.unique(ytr[tr_idx])) < 2:
        raise ValueError(f"Fold {fold} training partition has one class.")
    if len(np.unique(ytr[va_idx])) < 2:
        raise ValueError(
            f"Fold {fold} held-out partition has one class. "
            "Grouped CV needs a different split configuration."
        )


# ============================================================
# 14. HYPERPARAMETER SEARCH
# ============================================================

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
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
    n_iter=50,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True
)

search.fit(Xtr, ytr, groups=groups)
best_model = search.best_estimator_

print("\nBest grouped-CV F1:", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)


# ============================================================
# 15. OOF PROBABILITIES AND TRAINING-ONLY THRESHOLD
# ============================================================

oof_probs = cross_val_predict(
    clone(best_model),
    Xtr,
    ytr,
    groups=groups,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.01):
    pred = (oof_probs >= threshold).astype(np.int32)

    threshold_results.append({
        "threshold": float(threshold),
        "Precision": precision_score(ytr, pred, zero_division=0),
        "Recall": recall_score(ytr, pred, zero_division=0),
        "F1": f1_score(ytr, pred, zero_division=0),
    })

threshold_df = pd.DataFrame(threshold_results)
best_row = threshold_df.loc[threshold_df["F1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\nBest OOF threshold:", round(best_threshold, 3))
print("OOF precision:", round(best_row["Precision"], 4))
print("OOF recall:", round(best_row["Recall"], 4))
print("OOF F1:", round(best_row["F1"], 4))


# ============================================================
# 16. EVALUATION HELPERS
# ============================================================

def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)

    print(f"\n{name.upper()} RESULTS")
    print("Threshold:", round(threshold, 3))
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_true, predictions, zero_division=0), 4
    ))
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Cheating"],
        zero_division=0
    ))

    if len(np.unique(y_true)) == 2:
        print("ROC-AUC:", round(
            roc_auc_score(y_true, probabilities), 4
        ))
        print("Average Precision:", round(
            average_precision_score(y_true, probabilities), 4
        ))

    return predictions


def per_video_metrics(video_groups, y_true, y_pred):
    rows = []

    for video in np.unique(video_groups):
        idx = video_groups == video
        yt = y_true[idx]
        yp = y_pred[idx]

        rows.append({
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

    return pd.DataFrame(rows).sort_values(
        "F1"
    ).reset_index(drop=True)


# ============================================================
# 17. VALIDATION
# ============================================================

val_probs = best_model.predict_proba(Xv)[:, 1]
val_pred = evaluate_split(
    "Validation", yv, val_probs, best_threshold
)

validation_video_results = per_video_metrics(
    val_video_groups, yv, val_pred
)

print("\nPer-video validation results:")
display(validation_video_results.round(4))


# ============================================================
# 18. FINAL TEST
# Do not use test results to tune the model or threshold.
# ============================================================

test_probs = best_model.predict_proba(Xte)[:, 1]
test_pred = evaluate_split(
    "Test", yte, test_probs, best_threshold
)

test_video_results = per_video_metrics(
    test_video_groups, yte, test_pred
)

print("\nPer-video test results:")
display(test_video_results.round(4))


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature_index": np.arange(FUSION_DIM),
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 feature indices:")
display(importance_df.head(15).reset_index(drop=True).round(5))


# ============================================================
# 20. SAVE OUTPUTS
# ============================================================

threshold_df.to_csv(
    RESULTS_DIR / "training_oof_thresholds.csv",
    index=False
)

validation_video_results.to_csv(
    RESULTS_DIR / "validation_per_video.csv",
    index=False
)

test_video_results.to_csv(
    RESULTS_DIR / "test_per_video.csv",
    index=False
)

importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

summary = {
    "classifier": "XGBoost",
    "vision_features": VISION_DIM,
    "audio_features": AUDIO_DIM,
    "fusion_features": FUSION_DIM,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "grouped_cv_f1": float(search.best_score_),
    "selected_threshold_from_training_oof": best_threshold,
    "training_oof_precision": float(best_row["Precision"]),
    "training_oof_recall": float(best_row["Recall"]),
    "training_oof_f1": float(best_row["F1"]),
    "best_parameters": search.best_params_,
    "train_windows": int(len(X_train)),
    "validation_windows": int(len(X_val)),
    "test_windows": int(len(X_test)),
    "train_videos": int(len(np.unique(groups))),
    "validation_videos": int(len(np.unique(val_video_groups))),
    "test_videos": int(len(np.unique(test_video_groups))),
    "audio_alignment": {
        split: {
            "matched": int(audio_alignment_report[split]["matched"]),
            "missing": int(len(audio_alignment_report[split]["missing"])),
        }
        for split in SPLITS
    },
}

with open(RESULTS_DIR / "model_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nFinished.")
print("Fusion dimensions:", FUSION_DIM)
print(
    "Train/Validation/Test windows:",
    len(X_train), len(X_val), len(X_test)
)
print("Results saved to:", RESULTS_DIR)
