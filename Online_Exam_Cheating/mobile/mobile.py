# ============================================================
# 1. Imports and configuration
# ============================================================

import os
import sys
import pickle

import pandas as pd
import torch
from torch import nn, optim
from google.colab import drive


MOBILE_PATH = "/content/drive/MyDrive/Online_Exam_Cheating/mobile"
PROJECT_PATH = "/content/drive/MyDrive/Online_Exam_Cheating"

HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5
EPOCHS = 100
LEARNING_RATE = 0.001

WINDOW_SIZE = 10
STRIDE = 8
BATCH_SIZE = 32

FEATURE_DIM = 8
MODEL_INPUT_DIM = 7

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. Setup
# ============================================================

def setup_environment():
    drive.mount("/content/drive")

    if MOBILE_PATH not in sys.path:
        sys.path.append(MOBILE_PATH)

    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)


# ============================================================
# 3. Load PKL files
# ============================================================

def load_pkl_files():
    paths = []
    datas = []

    if not os.path.isdir(MOBILE_PATH):
        raise FileNotFoundError(
            f"MOBILE_PATH does not exist: {MOBILE_PATH}"
        )

    # Preserve the original directory-order behavior.
    # The resulting index-to-file mapping is printed below.
    for filename in os.listdir(MOBILE_PATH):
        if not filename.endswith(".pkl"):
            continue

        filepath = os.path.join(MOBILE_PATH, filename)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # This pipeline expects one record per timestep,
        # with an "objects" field in each record.
        if (
            not isinstance(data, list)
            or len(data) == 0
            or not isinstance(data[0], dict)
            or "objects" not in data[0]
        ):
            print(
                f"SKIPPED unsupported PKL schema: {filename}"
            )
            continue

        paths.append(filename)
        datas.append(data)

    if not datas:
        raise ValueError(
            "No compatible PKL files were found. "
            "Expected a list of timestep records containing "
            "an 'objects' field."
        )

    print("\nVideo index -> PKL filename")

    for i, filename in enumerate(paths):
        print(i, "->", filename)

    print("\nNumber of compatible videos:", len(datas))

    return paths, datas


# ============================================================
# 4. Extract mobile features
# ============================================================

def extract_features(paths, datas):
    all_features = []

    feature_keys = [
        "distance_wrist_left",
        "distance_wrist_right",
        "distance_finger_tip_left",
        "distance_finger_tip_right",
        "distance_thumb_tip_left",
        "distance_thumb_tip_right",
        "conf",
        "class_id",
    ]

    for filename, records in zip(paths, datas):

        feature_values = {
            key: [] for key in feature_keys
        }

        names = []

        for frame_index, record in enumerate(records):

            objects = record.get("objects")

            if not objects:
                values = {
                    "distance_wrist_left": 0.0,
                    "distance_wrist_right": 0.0,
                    "distance_finger_tip_left": 0.0,
                    "distance_finger_tip_right": 0.0,
                    "distance_thumb_tip_left": 0.0,
                    "distance_thumb_tip_right": 0.0,
                    "conf": 0.0,
                    "class_id": 7,
                }

                object_name = "unknown"

            else:
                best_object = max(
                    objects,
                    key=lambda obj: (
                        obj.get("conf")
                        if obj.get("conf") is not None
                        else 0.0
                    )
                )

                values = {}

                for key in feature_keys:
                    value = best_object.get(key)

                    if value is None:
                        value = 7 if key == "class_id" else 0.0

                    values[key] = value

                object_name = (
                    best_object.get("name") or "unknown"
                )

            for key in feature_keys:
                feature_values[key].append(
                    float(feature_values_value(values[key]))
                )

            names.append(object_name)

        tensors = {
            key: torch.tensor(
                values,
                dtype=torch.float32
            )
            for key, values in feature_values.items()
        }

        all_features.append({
            "video_id": filename,
            "tensor": tensors,
            "name": names,
        })

    return all_features


def feature_values_value(value):
    return value


# ============================================================
# 5. Group features
# ============================================================

def group_features(all_features):
    grouped_features = []

    for video in all_features:
        tensors = video["tensor"]

        wrist = torch.stack(
            [
                tensors["distance_wrist_left"],
                tensors["distance_wrist_right"],
            ],
            dim=1
        )

        finger = torch.stack(
            [
                tensors["distance_finger_tip_left"],
                tensors["distance_finger_tip_right"],
            ],
            dim=1
        )

        thumb = torch.stack(
            [
                tensors["distance_thumb_tip_left"],
                tensors["distance_thumb_tip_right"],
            ],
            dim=1
        )

        grouped_features.append({
            "wrist": wrist,
            "finger": finger,
            "thumb": thumb,
            "conf": tensors["conf"],
            "class_id": tensors["class_id"],
        })

    return grouped_features


# ============================================================
# 6. Build video representations
# ============================================================

def build_video_representations(
    all_features,
    grouped_features
):
    all_video_representations = []

    for features, grouped in zip(
        all_features,
        grouped_features
    ):
        T = torch.stack(
            [
                grouped["wrist"][:, 0],
                grouped["wrist"][:, 1],
                grouped["finger"][:, 0],
                grouped["finger"][:, 1],
                grouped["thumb"][:, 0],
                grouped["thumb"][:, 1],
                grouped["conf"],
                grouped["class_id"],
            ],
            dim=1
        )

        if T.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Unexpected feature dimension: {T.shape}"
            )

        all_video_representations.append(T)

        print(
            features["video_id"],
            "->",
            tuple(T.shape)
        )

    return all_video_representations


# ============================================================
# 7. Split videos
# ============================================================

def split_videos(all_video_representations):
    required_indices = [2, 3, 22, 6, 4, 5, 7, 8]

    if max(required_indices) >= len(
        all_video_representations
    ):
        raise ValueError(
            f"The split requires at least "
            f"{max(required_indices) + 1} compatible videos, "
            f"but only {len(all_video_representations)} "
            f"were loaded. Check the PKL files and their order."
        )

    validation_indices = [2, 3, 22, 6]
    test_indices = [4, 5, 7, 8]

    excluded_indices = set(
        validation_indices + test_indices
    )

    all_video_representations_validation = [
        all_video_representations[i]
        for i in validation_indices
    ]

    all_video_representations_test = [
        all_video_representations[i]
        for i in test_indices
    ]

    all_video_representations_train = [
        video
        for i, video in enumerate(
            all_video_representations
        )
        if i not in excluded_indices
    ]

    print("\nValidation video lengths:")
    for i, video in enumerate(
        all_video_representations_validation
    ):
        print(i, video.shape[0])

    print("\nTest video lengths:")
    for i, video in enumerate(
        all_video_representations_test
    ):
        print(i, video.shape[0])

    print(
        "\nTraining videos:",
        len(all_video_representations_train)
    )

    return (
        all_video_representations_train,
        all_video_representations_validation,
        all_video_representations_test
    )


# ============================================================
# 8. Create windows and masks
# ============================================================

def create_windows(videos):
    windows = []
    masks = []
    window_map = []

    for video_index, video in enumerate(videos):
        video_length = video.shape[0]

        for start in range(0, video_length, STRIDE):
            current_window = video[
                start:start + WINDOW_SIZE
            ]

            valid_length = current_window.shape[0]

            if valid_length == 0:
                continue

            if valid_length < WINDOW_SIZE:
                padding_length = WINDOW_SIZE - valid_length

                padding = torch.zeros(
                    padding_length,
                    video.shape[1],
                    dtype=video.dtype
                )

                current_window = torch.cat(
                    [current_window, padding],
                    dim=0
                )

            mask = torch.cat([
                torch.ones(valid_length),
                torch.zeros(WINDOW_SIZE - valid_length)
            ])

            windows.append(current_window)
            masks.append(mask)

            # Original video index and starting timestep.
            window_map.append((video_index, start))

    if not windows:
        raise ValueError("No windows were created.")

    windows_tensor = torch.stack(windows)
    masks_tensor = torch.stack(masks)

    assert windows_tensor.shape[:2] == masks_tensor.shape

    return windows_tensor, masks_tensor, window_map


# ============================================================
# 9. Normalize features
# ============================================================

def normalize_features(
    train_tensor,
    validation_tensor,
    test_tensor
):
    if train_tensor.shape[-1] != FEATURE_DIM:
        raise ValueError("Unexpected training feature dimension.")

    # Feature index 7 is class_id and is excluded from the model.
    feature_indices = list(range(7))

    train_min = train_tensor.amin(dim=(0, 1))
    train_max = train_tensor.amax(dim=(0, 1))

    normalized = []

    for tensor in [
        train_tensor,
        validation_tensor,
        test_tensor
    ]:
        selected = tensor[:, :, feature_indices]

        selected_min = train_min[feature_indices]
        selected_max = train_max[feature_indices]

        denominator = selected_max - selected_min

        safe_denominator = torch.where(
            denominator == 0,
            torch.ones_like(denominator),
            denominator
        )

        result = (
            selected - selected_min
        ) / safe_denominator

        # Avoid invalid values for constant features.
        constant_features = denominator == 0

        result[:, :, constant_features] = 0.0

        normalized.append(result)

    train_normalized, validation_normalized, test_normalized = (
        normalized
    )

    print("\nNormalized tensor shapes:")
    print("Train:", tuple(train_normalized.shape))
    print("Validation:", tuple(validation_normalized.shape))
    print("Test:", tuple(test_normalized.shape))

    return (
        train_normalized,
        validation_normalized,
        test_normalized
    )


# ============================================================
# 10. Dataset and DataLoaders
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, windows, masks):
        if len(windows) != len(masks):
            raise ValueError(
                "Window and mask counts do not match."
            )

        self.windows = windows
        self.masks = masks

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.masks[idx]


def create_data_loaders(
    train_tensor,
    validation_tensor,
    test_tensor,
    train_masks,
    validation_masks,
    test_masks
):
    train_dataset = SequenceDataset(
        train_tensor,
        train_masks
    )

    validation_dataset = SequenceDataset(
        validation_tensor,
        validation_masks
    )

    test_dataset = SequenceDataset(
        test_tensor,
        test_masks
    )

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

    return train_loader, validation_loader, test_loader


# ============================================================
# 11. Build model
# ============================================================

def build_model(input_dim):
    from models.gru import MultiLayerGRU

    model = MultiLayerGRU(
        input_dim=input_dim,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return model, optimizer


# ============================================================
# 12. Train model
# ============================================================

def train_model(model, optimizer, train_loader):
    epoch_losses = []

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []

        for batch_windows, batch_masks in train_loader:
            batch_windows = batch_windows.to(device)
            batch_masks = batch_masks.to(device).unsqueeze(-1)

            batch_size = batch_windows.shape[0]

            h0 = torch.zeros(
                NUM_LAYERS,
                batch_size,
                HIDDEN_SIZE,
                device=device
            )

            optimizer.zero_grad()

            reconstruction, _ = model(
                batch_windows,
                h0
            )

            squared_error = (
                batch_windows - reconstruction
            ) ** 2

            masked_error = squared_error * batch_masks

            denominator = (
                batch_masks.sum()
                * batch_windows.shape[-1]
            ).clamp_min(1)

            loss = masked_error.sum() / denominator

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {epoch_loss:.6f}"
        )

    return epoch_losses


# ============================================================
# 13. Calculate window and timestep errors
# ============================================================

def calculate_errors(model, loader):
    model.eval()

    window_errors = []
    timestep_errors = []
    all_masks = []
    total_squared_error = 0.0
    total_valid_values = 0.0

    with torch.no_grad():
        for windows, masks in loader:
            windows = windows.to(device)
            masks = masks.to(device)

            expanded_masks = masks.unsqueeze(-1)

            batch_size = windows.shape[0]

            h0 = torch.zeros(
                NUM_LAYERS,
                batch_size,
                HIDDEN_SIZE,
                device=device
            )

            reconstruction, _ = model(windows, h0)

            squared_error = (
                windows - reconstruction
            ) ** 2

            masked_error = squared_error * expanded_masks

            # One error for every timestep in each window.
            per_timestep_error = masked_error.mean(dim=2)

            timestep_errors.append(
                per_timestep_error.cpu()
            )

            all_masks.append(masks.cpu())

            valid_steps = masks.sum(dim=1)

            denominator = (
                valid_steps * windows.shape[-1]
            ).clamp_min(1)

            per_window_error = (
                masked_error.sum(dim=(1, 2))
                / denominator
            )

            window_errors.extend(
                per_window_error.cpu().tolist()
            )

            total_squared_error += masked_error.sum().item()

            total_valid_values += (
                masks.sum().item() * windows.shape[-1]
            )

    timestep_errors = torch.cat(
        timestep_errors,
        dim=0
    )

    all_masks = torch.cat(all_masks, dim=0)

    overall_loss = (
        total_squared_error / max(total_valid_values, 1)
    )

    print("Window errors:", len(window_errors))
    print("Timestep error shape:", tuple(timestep_errors.shape))
    print("Masked reconstruction loss:", overall_loss)

    return (
        window_errors,
        timestep_errors,
        all_masks,
        overall_loss
    )


# ============================================================
# 14. Create manual labels for validation and test
# ============================================================

def create_label_windows(videos, label_ranges):
    if len(videos) != len(label_ranges):
        raise ValueError(
            "Each video must have its own label-range list."
        )

    label_windows = []
    label_masks = []

    for video_index, (video, ranges) in enumerate(
        zip(videos, label_ranges)
    ):
        video_length = video.shape[0]

        labels = torch.zeros(
            video_length,
            dtype=torch.long
        )

        for start, end, label in ranges:
            if (
                start < 0
                or end < start
                or end >= video_length
            ):
                raise ValueError(
                    f"Invalid label range {(start, end, label)} "
                    f"for video {video_index}, "
                    f"whose actual length is {video_length}. "
                    "Check the selected video and its manual labels."
                )

            labels[start:end + 1] = label

        for start in range(0, video_length, STRIDE):
            current = labels[start:start + WINDOW_SIZE]
            valid_length = len(current)

            if valid_length == 0:
                continue

            if valid_length < WINDOW_SIZE:
                current = torch.cat([
                    current,
                    torch.zeros(
                        WINDOW_SIZE - valid_length,
                        dtype=torch.long
                    )
                ])

            mask = torch.cat([
                torch.ones(valid_length),
                torch.zeros(WINDOW_SIZE - valid_length)
            ])

            label_windows.append(current)
            label_masks.append(mask)

    if not label_windows:
        raise ValueError("No label windows were created.")

    label_windows = torch.stack(label_windows)
    label_masks = torch.stack(label_masks)

    return label_windows, label_masks


# ============================================================
# 15. Evaluate reconstruction errors by class
# ============================================================

def evaluate_class_errors(
    timestep_errors,
    error_masks,
    label_windows,
    label_masks,
    output_csv
):
    errors = timestep_errors.cpu()
    error_masks = error_masks.cpu().bool()

    labels = label_windows.cpu()
    label_masks = label_masks.cpu().bool()

    if errors.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: errors={tuple(errors.shape)}, "
            f"labels={tuple(labels.shape)}"
        )

    if error_masks.shape != label_masks.shape:
        raise ValueError(
            f"Mask shape mismatch: "
            f"error masks={tuple(error_masks.shape)}, "
            f"label masks={tuple(label_masks.shape)}"
        )

    if not torch.equal(error_masks, label_masks):
        raise ValueError(
            "Data masks and label masks are not aligned."
        )

    errors = errors.flatten()
    labels = labels.flatten()
    valid = error_masks.flatten()

    # Exclude padded positions.
    errors = errors[valid]
    labels = labels[valid]

    class_names = {
        0: "Normal",
        1: "Cheating: Book / Note / Paper",
        2: "Cheating: Using a Phone"
    }

    rows = []

    for label, name in class_names.items():
        class_errors = errors[labels == label]

        rows.append({
            "Class": name,
            "Label": label,
            "Number of valid window-timesteps": len(class_errors),
            "Mean reconstruction error": (
                class_errors.mean().item()
                if len(class_errors) > 0
                else float("nan")
            ),
            "Median reconstruction error": (
                class_errors.median().item()
                if len(class_errors) > 0
                else float("nan")
            ),
            "Standard deviation": (
                class_errors.std().item()
                if len(class_errors) > 1
                else float("nan")
            ),
            "Minimum reconstruction error": (
                class_errors.min().item()
                if len(class_errors) > 0
                else float("nan")
            ),
            "Maximum reconstruction error": (
                class_errors.max().item()
                if len(class_errors) > 0
                else float("nan")
            )
        })

    table = pd.DataFrame(rows)

    display(table)
    table.to_csv(output_csv, index=False)

    print("Saved:", output_csv)

    return table


# ============================================================
# 16. Save window-level test summary
# ============================================================

def save_test_summary(test_errors):
    series = pd.Series(test_errors)

    results = pd.DataFrame({
        "Component": [
            "Modality",
            "Model",
            "Input features",
            "Window size",
            "Stride",
            "Score level",
            "Number of test windows",
            "Mean reconstruction error",
            "Median reconstruction error",
            "Standard deviation",
            "Minimum reconstruction error",
            "Maximum reconstruction error",
            "Final use"
        ],
        "Description": [
            "Mobile",
            "MultiLayerGRU",
            (
                "Wrist, finger-tip and thumb-tip distances, "
                "plus confidence"
            ),
            WINDOW_SIZE,
            STRIDE,
            "Window-level anomaly score",
            len(test_errors),
            series.mean(),
            series.median(),
            series.std(),
            series.min(),
            series.max(),
            "Input to multimodal fusion"
        ]
    })

    display(results)

    results.to_csv(
        "mobile_test_summary_table.csv",
        index=False
    )

    return results


# ============================================================
# 17. Main pipeline
# ============================================================

def main():
    setup_environment()

    # Load
    paths, datas = load_pkl_files()

    # Extract and group features
    all_features = extract_features(paths, datas)

    grouped_features = group_features(all_features)

    all_video_representations = build_video_representations(
        all_features,
        grouped_features
    )

    # Split
    (
        train_videos,
        validation_videos,
        test_videos
    ) = split_videos(all_video_representations)

    # Create data windows
    train_tensor, train_masks, _ = create_windows(train_videos)

    validation_tensor, validation_masks, _ = create_windows(
        validation_videos
    )

    test_tensor, test_masks, _ = create_windows(test_videos)

    print("\nWindow counts:")
    print("Train:", len(train_tensor), len(train_masks))
    print(
        "Validation:",
        len(validation_tensor),
        len(validation_masks)
    )
    print("Test:", len(test_tensor), len(test_masks))

    # Normalize using training data only
    (
        train_normalized,
        validation_normalized,
        test_normalized
    ) = normalize_features(
        train_tensor,
        validation_tensor,
        test_tensor
    )

    # DataLoaders
    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_data_loaders(
        train_normalized,
        validation_normalized,
        test_normalized,
        train_masks,
        validation_masks,
        test_masks
    )

    # Model
    model, optimizer = build_model(
        input_dim=train_normalized.shape[-1]
    )

    # Training
    epoch_losses = train_model(
        model,
        optimizer,
        train_loader
    )

    # Validation errors
    (
        validation_window_errors,
        validation_timestep_errors,
        validation_error_masks,
        validation_loss
    ) = calculate_errors(
        model,
        validation_loader
    )

    # Test errors
    (
        test_window_errors,
        test_timestep_errors,
        test_error_masks,
        test_loss
    ) = calculate_errors(
        model,
        test_loader
    )

    print("\nFinal masked losses:")
    print("Validation:", validation_loss)
    print("Test:", test_loss)

    # --------------------------------------------------------
    # Validation labels
    # Ranges are inclusive: (start, end, label).
    # They must correspond to validation_videos in split order.
    # --------------------------------------------------------

    validation_label_ranges = [
        [],
        [],
        [
            (47, 64, 1),
            (246, 258, 1),
            (388, 420, 1),
            (423, 427, 1),
            (476, 524, 2),
            (687, 709, 1),
        ],
        [
            (514, 515, 1),
            (516, 532, 1),
        ],
    ]

    validation_label_windows, validation_label_masks = (
        create_label_windows(
            validation_videos,
            validation_label_ranges
        )
    )

    validation_table = evaluate_class_errors(
        validation_timestep_errors,
        validation_error_masks,
        validation_label_windows,
        validation_label_masks,
        "mobile_validation_label_evaluation.csv"
    )

    # --------------------------------------------------------
    # Test labels
    # Ranges must correspond to test_videos in split order.
    # --------------------------------------------------------

    test_label_ranges = [
        [
            (288, 307, 1),
            (688, 691, 1),
            (694, 695, 1),
        ],
        [
            (68, 71, 1),
            (74, 81, 1),
            (82, 83, 1),
            (85, 87, 1),
            (732, 735, 1),
            (904, 906, 1),
        ],
        [
            (428, 432, 1),
            (439, 441, 1),
            (444, 445, 1),
            (447, 450, 1),
            (453, 454, 1),
        ],
        [
            (556, 563, 1),
            (565, 566, 1),
            (567, 568, 1),
            (577, 585, 1),
            (607, 611, 1),
            (612, 616, 1),
            (691, 699, 1),
            (1247, 1258, 1),
            (1270, 1276, 1),
        ],
    ]

    test_label_windows, test_label_masks = create_label_windows(
        test_videos,
        test_label_ranges
    )

    test_label_table = evaluate_class_errors(
        test_timestep_errors,
        test_error_masks,
        test_label_windows,
        test_label_masks,
        "mobile_test_label_evaluation.csv"
    )

    # Window-level test summary
    test_summary = save_test_summary(test_window_errors)

    # Save training history
    pd.DataFrame({
        "Epoch": range(1, len(epoch_losses) + 1),
        "Training loss": epoch_losses
    }).to_csv(
        "mobile_training_history.csv",
        index=False
    )

    print("\nPipeline completed.")
    print("Generated CSV files:")
    print("- mobile_validation_label_evaluation.csv")
    print("- mobile_test_label_evaluation.csv")
    print("- mobile_test_summary_table.csv")
    print("- mobile_training_history.csv")

    return {
        "model": model,
        "epoch_losses": epoch_losses,
        "validation_table": validation_table,
        "test_label_table": test_label_table,
        "test_summary": test_summary,
    }


# ============================================================
# 18. Run
# ============================================================

if __name__ == "__main__":
    results = main()
