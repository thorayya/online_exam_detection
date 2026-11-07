
# ============================================================
# Mobile Self-Attention Data Preprocessing
# ============================================================
#
# Description:
# Extracts mobile-related features from PKL files, creates
# video representations, splits videos into train/validation/
# test sets, generates temporal windows and masks, normalizes
# model inputs, and saves all data with temporal metadata.
#
# Feature configuration:
# - Raw features: 8
# - Model input features: 7
# - Window size: 10 seconds
# - Stride: 8 seconds
# - Sampling interval: 25 frames (1 second at 25 FPS)
#
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from google.colab import drive


# ============================================================
# 2. Configuration
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/Online_Exam_Cheating"
MOBILE_DIR = os.path.join(PROJECT_DIR, "mobile")

OUTPUT_PATH = os.path.join(
    MOBILE_DIR,
    "mobile_self_attention_data.pt"
)

WINDOW_SIZE = 10
STRIDE = 8
BATCH_SIZE = 32

SOURCE_FPS = 25
SAMPLING_INTERVAL_FRAMES = 25
SECONDS_PER_ROW = 1

FEATURE_NAMES = [
    "distance_wrist_left",
    "distance_wrist_right",
    "distance_finger_tip_left",
    "distance_finger_tip_right",
    "distance_thumb_tip_left",
    "distance_thumb_tip_right",
    "conf",
    "class_id"
]

MODEL_FEATURE_NAMES = FEATURE_NAMES[:7]


# ============================================================
# 3. Mount Google Drive
# ============================================================

drive.mount("/content/drive")


# ============================================================
# 4. Safe value conversion
# ============================================================

def safe_float(value, default=0.0):

    if value is None:
        return default

    try:
        value = float(value)

        if not torch.isfinite(torch.tensor(value)):
            return default

        return value

    except (TypeError, ValueError):
        return default


# ============================================================
# 5. Load PKL files
# ============================================================

paths = []

for path in os.listdir(MOBILE_DIR):

    if path.endswith(".pkl"):
        paths.append(path)


print("Number of PKL files:", len(paths))


datas = []

for i in range(len(paths)):

    with open(
        os.path.join(MOBILE_DIR, paths[i]),
        "rb"
    ) as f:

        datas.append(pickle.load(f))


# ============================================================
# 6. Extract mobile features
# ============================================================

all_features = []

for data in range(len(datas)):

    frames = datas[data]

    if len(frames) == 0:
        print("Empty video:", paths[data])
        continue

    distance_wrist_right = []
    distance_wrist_left = []

    distance_finger_tip_right = []
    distance_finger_tip_left = []

    distance_thumb_tip_right = []
    distance_thumb_tip_left = []

    conf = []
    name = []
    class_id = []

    features = {
        "video_id": frames[0].get("video", paths[data]),
        "file_name": paths[data],
        "tensor": [],
        "name": name
    }

    for j in range(len(frames)):

        objects = frames[j].get("objects")

        if objects is None or len(objects) == 0:

            distance_wrist_left.append(0)
            distance_wrist_right.append(0)

            distance_finger_tip_left.append(0)
            distance_finger_tip_right.append(0)

            distance_thumb_tip_left.append(0)
            distance_thumb_tip_right.append(0)

            conf.append(0)
            class_id.append(7)
            name.append("unknown")

        else:

            best_object = max(
                objects,
                key=lambda obj: safe_float(obj.get("conf"))
            )

            distance_wrist_left.append(
                safe_float(
                    best_object.get("distance_wrist_left")
                )
            )

            distance_wrist_right.append(
                safe_float(
                    best_object.get("distance_wrist_right")
                )
            )

            distance_finger_tip_left.append(
                safe_float(
                    best_object.get("distance_finger_tip_left")
                )
            )

            distance_finger_tip_right.append(
                safe_float(
                    best_object.get("distance_finger_tip_right")
                )
            )

            distance_thumb_tip_left.append(
                safe_float(
                    best_object.get("distance_thumb_tip_left")
                )
            )

            distance_thumb_tip_right.append(
                safe_float(
                    best_object.get("distance_thumb_tip_right")
                )
            )

            conf.append(
                safe_float(best_object.get("conf"))
            )

            class_id.append(
                safe_float(
                    best_object.get("class_id"),
                    7
                )
            )

            name.append(
                best_object.get("name")
                if best_object.get("name") is not None
                else "unknown"
            )

    distance_wrist_left = torch.tensor(
        distance_wrist_left,
        dtype=torch.float32
    )

    distance_wrist_right = torch.tensor(
        distance_wrist_right,
        dtype=torch.float32
    )

    distance_finger_tip_left = torch.tensor(
        distance_finger_tip_left,
        dtype=torch.float32
    )

    distance_finger_tip_right = torch.tensor(
        distance_finger_tip_right,
        dtype=torch.float32
    )

    distance_thumb_tip_left = torch.tensor(
        distance_thumb_tip_left,
        dtype=torch.float32
    )

    distance_thumb_tip_right = torch.tensor(
        distance_thumb_tip_right,
        dtype=torch.float32
    )

    conf = torch.tensor(conf, dtype=torch.float32)

    class_id = torch.tensor(class_id, dtype=torch.float32)

    features["tensor"].append(distance_wrist_left)
    features["tensor"].append(distance_wrist_right)

    features["tensor"].append(distance_finger_tip_left)
    features["tensor"].append(distance_finger_tip_right)

    features["tensor"].append(distance_thumb_tip_left)
    features["tensor"].append(distance_thumb_tip_right)

    features["tensor"].append(conf)
    features["tensor"].append(class_id)

    features["name"] = name

    all_features.append(features)


# ============================================================
# 7. Group features
# ============================================================

grouped_features = []

for i in range(len(all_features)):

    wrist = torch.stack(
        all_features[i]["tensor"][0:2],
        dim=1
    )

    finger = torch.stack(
        all_features[i]["tensor"][2:4],
        dim=1
    )

    thumb = torch.stack(
        all_features[i]["tensor"][4:6],
        dim=1
    )

    conf = all_features[i]["tensor"][6]

    class_id = all_features[i]["tensor"][7]

    grouped_features.append({

        "wrist": wrist,
        "finger": finger,
        "thumb": thumb,
        "conf": conf,
        "class_id": class_id

    })


# ============================================================
# 8. Create video representations
# ============================================================

all_video_representations = []

for i in range(len(all_features)):

    num_frames = int(
        all_features[i]["tensor"][0].shape[0]
    )

    distance_wrist_left = grouped_features[i]["wrist"][:, 0]
    distance_wrist_right = grouped_features[i]["wrist"][:, 1]

    distance_finger_tip_left = grouped_features[i]["finger"][:, 0]
    distance_finger_tip_right = grouped_features[i]["finger"][:, 1]

    distance_thumb_tip_left = grouped_features[i]["thumb"][:, 0]
    distance_thumb_tip_right = grouped_features[i]["thumb"][:, 1]

    conf = grouped_features[i]["conf"]
    class_id = grouped_features[i]["class_id"]

    T = torch.stack([

        distance_wrist_left,
        distance_wrist_right,

        distance_finger_tip_left,
        distance_finger_tip_right,

        distance_thumb_tip_left,
        distance_thumb_tip_right,

        conf,
        class_id

    ], dim=1)

    all_video_representations.append(T)


print("\nTotal videos:", len(all_video_representations))


# ============================================================
# 9. Train / Validation / Test split
# ============================================================

all_video_representations_validation = [
    all_video_representations[11],
    all_video_representations[15],
    all_video_representations[9],
    all_video_representations[12]
]

all_video_representations_test = [
    all_video_representations[16],
    all_video_representations[14],
    all_video_representations[20],
    all_video_representations[13]
]

all_video_representations_train = (

    all_video_representations[:9]

    + [all_video_representations[10]]

    + all_video_representations[17:20]

    + all_video_representations[21:]

)


train_indices = (
    list(range(9))
    + [10]
    + list(range(17, 20))
    + list(range(21, len(all_video_representations)))
)

validation_indices = [11, 15, 9, 12]

test_indices = [16, 14, 20, 13]


# Check split

all_indices = (
    train_indices
    + validation_indices
    + test_indices
)

assert len(all_indices) == len(set(all_indices)), (
    "Train, validation and test overlap."
)

assert set(all_indices) == set(
    range(len(all_video_representations))
), "Some video indices are missing or invalid."


print("\nTrain videos:", len(all_video_representations_train))
print("Validation videos:", len(all_video_representations_validation))
print("Test videos:", len(all_video_representations_test))


# ============================================================
# 10. Create windows and masks
# ============================================================

def create_video_windows(video_indices, split_name):

    windows = []
    masks = []
    metadata = []

    for video_index in video_indices:

        video = all_video_representations[video_index]

        video_id = all_features[video_index]["video_id"]
        file_name = all_features[video_index]["file_name"]

        for j in range(0, video.shape[0], STRIDE):

            current_window = video[j:j + WINDOW_SIZE]

            valid_length = current_window.shape[0]

            if valid_length == 0:
                continue

            if valid_length < WINDOW_SIZE:

                padding_length = WINDOW_SIZE - valid_length

                padding = torch.zeros(
                    padding_length,
                    8,
                    dtype=torch.float32
                )

                current_window = torch.cat(
                    [current_window, padding],
                    dim=0
                )

                mask = torch.cat([

                    torch.ones(valid_length),

                    torch.zeros(padding_length)

                ])

            else:

                mask = torch.ones(WINDOW_SIZE)

            windows.append(current_window)
            masks.append(mask)

            metadata.append({

                "window_id": len(metadata),

                "video_index": video_index,

                "video_id": video_id,

                "file_name": file_name,

                "split": split_name,

                "start_sample_index": j,

                "end_sample_index": j + valid_length,

                "start_time_seconds": float(j),

                "end_time_seconds": float(j + valid_length),

                "window_size_seconds": WINDOW_SIZE,

                "valid_seconds": valid_length

            })

    return (
        torch.stack(windows),
        torch.stack(masks),
        metadata
    )


# ============================================================
# 11. Create train / validation / test windows
# ============================================================

train_tensor, masks, train_metadata = create_video_windows(
    train_indices,
    "train"
)

validation_tensor, masks_validation, validation_metadata = create_video_windows(
    validation_indices,
    "validation"
)

test_tensor, masks_test, test_metadata = create_video_windows(
    test_indices,
    "test"
)


print("\nRaw window shapes:")

print("Train:", train_tensor.shape)
print("Validation:", validation_tensor.shape)
print("Test:", test_tensor.shape)

print("\nMask shapes:")

print("Train:", masks.shape)
print("Validation:", masks_validation.shape)
print("Test:", masks_test.shape)


# ============================================================
# 12. Calculate normalization values
#     Only valid train positions are used.
# ============================================================

valid_train_values = train_tensor[masks.bool()]

feature_max = valid_train_values.amax(dim=0)

feature_min = valid_train_values.amin(dim=0)


for i in range(8):

    print(
        f"Feature {i}: "
        f"max = {feature_max[i].item():.4f}, "
        f"min = {feature_min[i].item():.4f}"
    )


# ============================================================
# 13. Normalize features
# ============================================================

normalized_features = []
normalized_validation_features = []
normalized_test_features = []

for i in range(8):

    if i == 7:
        continue

    feature_range = feature_max[i] - feature_min[i]

    if feature_range == 0:
        feature_range = torch.tensor(1.0)

    z = (
        train_tensor[:, :, i] - feature_min[i]
    ) / feature_range

    z_validation = (
        validation_tensor[:, :, i] - feature_min[i]
    ) / feature_range

    z_test = (
        test_tensor[:, :, i] - feature_min[i]
    ) / feature_range

    normalized_features.append(z)

    normalized_validation_features.append(z_validation)

    normalized_test_features.append(z_test)


normalized_features = torch.stack(normalized_features)

normalized_validation_features = torch.stack(
    normalized_validation_features
)

normalized_test_features = torch.stack(
    normalized_test_features
)


normalized_features = normalized_features.permute(1, 2, 0)

normalized_validation_features = (
    normalized_validation_features.permute(1, 2, 0)
)

normalized_test_features = (
    normalized_test_features.permute(1, 2, 0)
)


# Keep padded positions equal to zero

normalized_features = (
    normalized_features * masks.unsqueeze(-1)
)

normalized_validation_features = (
    normalized_validation_features
    * masks_validation.unsqueeze(-1)
)

normalized_test_features = (
    normalized_test_features
    * masks_test.unsqueeze(-1)
)


# ============================================================
# 14. Check normalized data
# ============================================================

print("\nNormalized feature ranges:")

print("Train min:", normalized_features.min().item())
print("Train max:", normalized_features.max().item())

print(
    "Validation min:",
    normalized_validation_features.min().item()
)

print(
    "Validation max:",
    normalized_validation_features.max().item()
)

print(
    "Validation shape:",
    normalized_validation_features.shape
)

print(
    "Test min:",
    normalized_test_features.min().item()
)

print(
    "Test max:",
    normalized_test_features.max().item()
)

print(
    "Test shape:",
    normalized_test_features.shape
)


# ============================================================
# 15. Sequence Dataset
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, windows, masks):

        self.windows = windows
        self.masks = masks

    def __len__(self):

        return len(self.windows)

    def __getitem__(self, idx):

        return self.windows[idx], self.masks[idx]


train_dataset = SequenceDataset(
    normalized_features,
    masks
)

validation_dataset = SequenceDataset(
    normalized_validation_features,
    masks_validation
)

test_dataset = SequenceDataset(
    normalized_test_features,
    masks_test
)


# ============================================================
# 16. DataLoaders
# ============================================================

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 17. Save all data and metadata
# ============================================================

mobile_data = {

    "feature_names": FEATURE_NAMES,

    "model_feature_names": MODEL_FEATURE_NAMES,

    "source_fps": SOURCE_FPS,

    "sampling_interval_frames": SAMPLING_INTERVAL_FRAMES,

    "seconds_per_row": SECONDS_PER_ROW,

    "window_size": WINDOW_SIZE,

    "stride": STRIDE,

    "batch_size": BATCH_SIZE,

    "video_features": [

        {
            "video_id": all_features[i]["video_id"],

            "file_name": all_features[i]["file_name"],

            "features": all_video_representations[i],

            "names": all_features[i]["name"]

        }

        for i in range(len(all_features))
    ],

    "train": {

        "raw_windows": train_tensor,

        "normalized_windows": normalized_features,

        "masks": masks,

        "metadata": train_metadata

    },

    "validation": {

        "raw_windows": validation_tensor,

        "normalized_windows": normalized_validation_features,

        "masks": masks_validation,

        "metadata": validation_metadata

    },

    "test": {

        "raw_windows": test_tensor,

        "normalized_windows": normalized_test_features,

        "masks": masks_test,

        "metadata": test_metadata

    },

    "normalization": {

        "feature_min": feature_min,

        "feature_max": feature_max,

        "normalized_feature_indices": list(range(7)),

        "excluded_feature": "class_id"

    },

    "split_indices": {

        "train": train_indices,

        "validation": validation_indices,

        "test": test_indices

    },

    "source_file_order": paths

}


torch.save(mobile_data, OUTPUT_PATH)


# ============================================================
# 18. Final checks
# ============================================================

print("\n========== FINAL RESULTS ==========")

print("Total videos:", len(all_video_representations))

print("Train windows:", len(train_dataset))

print("Validation windows:", len(validation_dataset))

print("Test windows:", len(test_dataset))

print("Train batches:", len(train_loader))

print("Validation batches:", len(validation_loader))

print("Test batches:", len(test_loader))

print("Model input dimension:", normalized_features.shape[-1])

print("Train input shape:", normalized_features.shape)

print("Validation input shape:", normalized_validation_features.shape)

print("Test input shape:", normalized_test_features.shape)

print("\nSaved file:")
print(OUTPUT_PATH)

print("\nMobile Self-Attention preprocessing completed.")
