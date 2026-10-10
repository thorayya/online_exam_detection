
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


MOBILE_PATH = ""
PROJECT_PATH = ""

HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5
EPOCHS = 100

WINDOW_SIZE = 10
STRIDE = 8
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Setup environment
# ============================================================

def setup_environment():
    drive.mount("/content/drive")

    sys.path.append(MOBILE_PATH)

    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)


# ============================================================
# 3. Load PKL files
# ============================================================

def load_pkl_files():
    paths = []

    # Keep the original file ordering behavior unchanged.
    for path in os.listdir(MOBILE_PATH):
        if path.endswith("pkl"):
            paths.append(path)

    datas = []

    for i in range(len(paths)):
        with open(f"{MOBILE_PATH}/{paths[i]}", "rb") as f:
            datas.append(pickle.load(f))

    return paths, datas


# ============================================================
# 4. Extract mobile features
# ============================================================

def extract_features(datas):
    all_features = []

    for data in range(len(datas)):
        frames = datas[data]

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
            "video_id": frames[0]["video"],
            "tensor": [],
            "name": name
        }

        for j in range(len(frames)):
            objects = frames[j]["objects"]

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
                    key=lambda obj: obj["conf"]
                )

                distance_wrist_left.append(
                    best_object["distance_wrist_left"]
                    if best_object["distance_wrist_left"] is not None
                    else 0
                )

                distance_wrist_right.append(
                    best_object["distance_wrist_right"]
                    if best_object["distance_wrist_right"] is not None
                    else 0
                )

                distance_finger_tip_left.append(
                    best_object["distance_finger_tip_left"]
                    if best_object["distance_finger_tip_left"] is not None
                    else 0
                )

                distance_finger_tip_right.append(
                    best_object["distance_finger_tip_right"]
                    if best_object["distance_finger_tip_right"] is not None
                    else 0
                )

                distance_thumb_tip_left.append(
                    best_object["distance_thumb_tip_left"]
                    if best_object["distance_thumb_tip_left"] is not None
                    else 0
                )

                distance_thumb_tip_right.append(
                    best_object["distance_thumb_tip_right"]
                    if best_object["distance_thumb_tip_right"] is not None
                    else 0
                )

                conf.append(
                    best_object["conf"]
                    if best_object["conf"] is not None
                    else 0
                )

                class_id.append(
                    best_object["class_id"]
                    if best_object["class_id"] is not None
                    else 7
                )

                name.append(
                    best_object["name"]
                    if best_object["name"] is not None
                    else "unknown"
                )

        distance_wrist_left = torch.tensor(distance_wrist_left)
        distance_wrist_right = torch.tensor(distance_wrist_right)

        distance_finger_tip_left = torch.tensor(
            distance_finger_tip_left
        )
        distance_finger_tip_right = torch.tensor(
            distance_finger_tip_right
        )

        distance_thumb_tip_left = torch.tensor(
            distance_thumb_tip_left
        )
        distance_thumb_tip_right = torch.tensor(
            distance_thumb_tip_right
        )

        conf = torch.tensor(conf)
        class_id = torch.tensor(class_id)

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

    return all_features


# ============================================================
# 5. Group features
# ============================================================

def group_features(all_features):
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

    return grouped_features


# ============================================================
# 6. Build video representations
# ============================================================

def build_video_representations(all_features, grouped_features):
    all_video_representations = []

    for i in range(len(all_features)):
        distance_wrist_left = grouped_features[i]["wrist"][:, 0]
        distance_wrist_right = grouped_features[i]["wrist"][:, 1]

        distance_finger_tip_left = grouped_features[i]["finger"][:, 0]
        distance_finger_tip_right = grouped_features[i]["finger"][:, 1]

        distance_thumb_tip_left = grouped_features[i]["thumb"][:, 0]
        distance_thumb_tip_right = grouped_features[i]["thumb"][:, 1]

        conf = grouped_features[i]["conf"]
        class_id = grouped_features[i]["class_id"]

        T = torch.stack(
            [
                distance_wrist_left,
                distance_wrist_right,
                distance_finger_tip_left,
                distance_finger_tip_right,
                distance_thumb_tip_left,
                distance_thumb_tip_right,
                conf,
                class_id
            ],
            dim=1
        )

        all_video_representations.append(T)

    print(len(all_video_representations))

    return all_video_representations


# ============================================================
# 7. Split videos into train, validation and test
# ============================================================

def split_videos(all_video_representations):
    all_video_representations_validation = [
        all_video_representations[2],
        all_video_representations[3],
        all_video_representations[22],
        all_video_representations[6]
    ]

    all_video_representations_test = [
        all_video_representations[4],
        all_video_representations[5],
        all_video_representations[7],
        all_video_representations[8]
    ]

    all_video_representations = (
        all_video_representations[:2]
        + all_video_representations[9:22]
        + all_video_representations[23:]
    )

    print(len(all_video_representations_validation))

    for i, video in enumerate(all_video_representations_validation):
        print(i, video.shape)

    print(len(all_video_representations_test))
    print(len(all_video_representations))

    return (
        all_video_representations,
        all_video_representations_validation,
        all_video_representations_test
    )


# ============================================================
# 8. Create validation windows
# ============================================================

def create_validation_windows(
    all_video_representations_validation
):
    windows_validation = []
    window_validation = WINDOW_SIZE
    stride_validation = STRIDE

    masks_validation = []
    video_index = []

    for i, video in enumerate(all_video_representations_validation):
        for j in range(0, video.shape[0], stride_validation):
            current_window = video[j:j + window_validation]

            if current_window.shape[0] == WINDOW_SIZE:
                windows_validation.append(current_window)

                mask = torch.ones(current_window.shape[0])
                masks_validation.append(mask)

            if current_window.shape[0] < WINDOW_SIZE:
                padding_length = window_validation - (
                    video.shape[0] - j
                )

                padding = torch.zeros(padding_length, 8)

                mask = torch.ones(current_window.shape[0])
                mask_padding = torch.zeros(padding_length)

                ending_window_validation = torch.cat(
                    [current_window, padding],
                    dim=0
                )

                ending_mask_validation = torch.cat(
                    [mask, mask_padding],
                    dim=0
                )

                masks_validation.append(ending_mask_validation)
                windows_validation.append(ending_window_validation)

                for k in range(len(ending_mask_validation)):
                    if ending_mask_validation[k] == 0:
                        video_index.append(
                            (len(windows_validation) - 1, i, k + j)
                        )
                        break

    window = torch.stack(windows_validation)

    return windows_validation, masks_validation, video_index


# ============================================================
# 9. Create test windows
# ============================================================

def create_test_windows(all_video_representations_test):
    windows_test = []
    masks_test = []

    window_test = WINDOW_SIZE
    stride_test = STRIDE

    for video in all_video_representations_test:
        for j in range(0, video.shape[0], stride_test):
            current_window = video[j:j + window_test]

            # Full window
            if current_window.shape[0] == window_test:
                windows_test.append(current_window)

                mask = torch.ones(window_test)
                masks_test.append(mask)

            # Final incomplete window
            elif (
                current_window.shape[0] < window_test
                and current_window.shape[0] > 0
            ):
                padding_length = window_test - current_window.shape[0]

                padding = torch.zeros(padding_length, 8)

                ending_window_test = torch.cat(
                    [current_window, padding],
                    dim=0
                )

                mask = torch.ones(current_window.shape[0])
                mask_padding_test = torch.zeros(padding_length)

                ending_mask_test = torch.cat(
                    [mask, mask_padding_test],
                    dim=0
                )

                windows_test.append(ending_window_test)
                masks_test.append(ending_mask_test)

    return windows_test, masks_test


# ============================================================
# 10. Create training windows
# ============================================================

def create_training_windows(all_video_representations):
    windows = []
    window = WINDOW_SIZE
    stride = STRIDE
    masks = []

    for i, video in enumerate(all_video_representations):
        for j in range(0, video.shape[0], stride):
            current_window = video[j:j + window]

            if current_window.shape[0] == WINDOW_SIZE:
                windows.append(current_window)

                mask = torch.ones(current_window.shape[0])
                masks.append(mask)

            if current_window.shape[0] < WINDOW_SIZE:
                padding_length = window - (video.shape[0] - j)

                padding = torch.zeros(padding_length, 8)

                mask = torch.ones(current_window.shape[0])
                mask_padding = torch.zeros(padding_length)

                ending_window = torch.cat(
                    [current_window, padding],
                    dim=0
                )

                ending_mask = torch.cat(
                    [mask, mask_padding],
                    dim=0
                )

                masks.append(ending_mask)
                windows.append(ending_window)

    print(len(windows))
    print(len(windows))
    print(windows[0].shape)

    return windows, masks


# ============================================================
# 11. Normalize features
# ============================================================

def normalize_features(
    windows,
    windows_validation,
    windows_test
):
    train_tensor = torch.stack(windows)
    validation_tensor = torch.stack(windows_validation)
    test_tensor = torch.stack(windows_test)

    normalized_features = []
    normalized_validation_features = []
    normalized_test_features = []

    feature_max = train_tensor.amax(dim=(0, 1))
    feature_min = train_tensor.amin(dim=(0, 1))

    for i in range(8):
        print(
            f"Feature {i}: "
            f"max = {feature_max[i].item():.4f}, "
            f"min = {feature_min[i].item():.4f}"
        )

        if i == 7:
            continue

        denominator = feature_max[i] - feature_min[i]

        if denominator == 0:
            z = torch.zeros_like(train_tensor[:, :, i])

            z_validation = torch.zeros_like(
                validation_tensor[:, :, i]
            )

            z_test = torch.zeros_like(test_tensor[:, :, i])

        else:
            z = (
                train_tensor[:, :, i] - feature_min[i]
            ) / denominator

            z_validation = (
                validation_tensor[:, :, i] - feature_min[i]
            ) / denominator

            z_test = (
                test_tensor[:, :, i] - feature_min[i]
            ) / denominator

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

    normalized_test_features = normalized_test_features.permute(
        1, 2, 0
    )

    print(normalized_features.min().item())
    print(normalized_features.max().item())

    print(
        f" valid min: "
        f"{normalized_validation_features.min().item()}"
    )
    print(
        f" valid max: "
        f"{normalized_validation_features.max().item()}"
    )
    print(normalized_validation_features.shape)

    print(
        f" test min: "
        f"{normalized_test_features.min().item()}"
    )
    print(
        f" test max: "
        f"{normalized_test_features.max().item()}"
    )
    print(normalized_test_features.shape)

    return (
        normalized_features,
        normalized_validation_features,
        normalized_test_features
    )


# ============================================================
# 12. Dataset class
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, windows, masks):
        self.windows = windows
        self.masks = masks

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.masks[idx]


# ============================================================
# 13. Create DataLoaders
# ============================================================

def create_data_loaders(
    normalized_features,
    normalized_validation_features,
    normalized_test_features,
    masks,
    masks_validation,
    masks_test
):
    train_dataset = SequenceDataset(
        normalized_features,
        masks
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validatation_dataset = SequenceDataset(
        normalized_validation_features,
        masks_validation
    )

    validatation_loader = torch.utils.data.DataLoader(
        validatation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = SequenceDataset(
        normalized_test_features,
        masks_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, validatation_loader, test_loader


# ============================================================
# 14. Build the GRU model
# ============================================================

def build_model(input_dim):
    from models.gru import MultiLayerGRU

    model = MultiLayerGRU(
        input_dim=input_dim,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return model, optimizer


# ============================================================
# 15. Train the model
# ============================================================

def train_model(model, optimizer, train_loader, input_dim):
    epoch_losses = []

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []

        for batch_windows, batch_masks in train_loader:
            batch_windows = batch_windows.to(device)
            batch_masks = batch_masks.to(device)

            batch_masks = batch_masks.unsqueeze(-1)

            batch_size = batch_windows.shape[0]

            h0 = torch.zeros(
                NUM_LAYERS,
                batch_size,
                HIDDEN_SIZE,
                device=device
            )

            optimizer.zero_grad()

            reconstruction, _ = model(batch_windows, h0)

            error = (batch_windows - reconstruction) ** 2
            masked_error = error * batch_masks

            loss = masked_error.sum() / (
                batch_masks.sum() * input_dim
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Loss: {epoch_loss:.6f}"
        )

    return epoch_losses, batch_windows, batch_masks, h0


# ============================================================
# 16. Run batch diagnostics
# ============================================================

def run_batch_diagnostics(
    model,
    batch_windows,
    batch_masks,
    h0,
    validatation_loader,
    test_loader
):
    print("batch_windows:", batch_windows.shape)
    print("batch_masks:", batch_masks.shape)
    print("h0:", h0.shape)

    reconstruction, _ = model(batch_windows, h0)

    error = (batch_windows - reconstruction) ** 2
    masked_error = error * batch_masks

    loss = torch.sum(masked_error) / (
        torch.sum(batch_masks) * batch_windows.shape[-1]
    )

    # Validation batch
    (
        batch_windows_validation,
        batch_masks_validation
    ) = next(iter(validatation_loader))

    batch_windows_validation = batch_windows_validation.to(device)
    batch_masks_validation = batch_masks_validation.to(device)

    batch_masks_validation = batch_masks_validation.unsqueeze(-1)

    h0_validation = torch.zeros(
        NUM_LAYERS,
        batch_windows_validation.shape[0],
        HIDDEN_SIZE,
        device=device
    )

    reconstruction_validation, _ = model(
        batch_windows_validation,
        h0_validation
    )

    error_validation = (
        batch_windows_validation - reconstruction_validation
    ) ** 2

    masked_error_validation = error_validation * batch_masks_validation

    loss_validation = torch.sum(masked_error_validation) / (
        torch.sum(batch_masks_validation)
        * batch_windows_validation.shape[-1]
    )

    # Test batch
    batch_windows_test, batch_masks_test = next(iter(test_loader))

    batch_windows_test = batch_windows_test.to(device)
    batch_masks_test = batch_masks_test.to(device)

    batch_masks_test = batch_masks_test.unsqueeze(-1)

    h0_test = torch.zeros(
        NUM_LAYERS,
        batch_windows_test.shape[0],
        HIDDEN_SIZE,
        device=device
    )

    reconstruction_test, _ = model(
        batch_windows_test,
        h0_test
    )

    error_test = (batch_windows_test - reconstruction_test) ** 2
    masked_error_test = error_test * batch_masks_test

    loss_test = torch.sum(masked_error_test) / (
        torch.sum(batch_masks_test)
        * batch_windows_test.shape[-1]
    )

    return loss, loss_validation, loss_test


# ============================================================
# 17. Calculate validation errors
# ============================================================

def calculate_validation_errors(model, validatation_loader):
    model.eval()

    batch_losses = []
    validation_errors = []
    timestep_error = []

    with torch.no_grad():
        for (
            batch_windows_validation,
            batch_masks_validation
        ) in validatation_loader:

            batch_windows_validation = (
                batch_windows_validation.to(device)
            )

            batch_masks_validation = (
                batch_masks_validation.to(device)
            )

            batch_masks_validation = (
                batch_masks_validation.unsqueeze(-1)
            )

            batch_size = batch_windows_validation.shape[0]

            h0 = torch.zeros(
                NUM_LAYERS,
                batch_size,
                HIDDEN_SIZE,
                device=device
            )

            reconstruction_validation, _ = model(
                batch_windows_validation,
                h0
            )

            error_validation = (
                batch_windows_validation - reconstruction_validation
            ) ** 2

            masked_error_validation = (
                error_validation * batch_masks_validation
            )

            timestep_error.append(
                masked_error_validation.mean(dim=2)
            )

            window_error = (
                torch.sum(masked_error_validation, dim=(1, 2))
                / (
                    torch.sum(batch_masks_validation, dim=(1, 2))
                    * batch_windows_validation.shape[-1]
                )
            )

            loss_validation = (
                torch.sum(masked_error_validation)
                / (
                    torch.sum(batch_masks_validation)
                    * batch_windows_validation.shape[-1]
                )
            )

            batch_losses.append(loss_validation.item())

            validation_errors.extend(
                window_error.cpu().tolist()
            )

    timestep_errors_validation = torch.cat(
        timestep_error,
        dim=0
    )

    print(timestep_errors_validation.shape)
    print("Number of validation windows:", len(validation_errors))
    print("Validation Loss:", sum(batch_losses) / len(batch_losses))

    return validation_errors, timestep_errors_validation


# ============================================================
# 18. Calculate test errors
# ============================================================

def calculate_test_errors(model, test_loader):
    test_errors = []

    model.eval()

    with torch.no_grad():
        for batch_windows_test, batch_masks_test in test_loader:
            batch_windows_test = batch_windows_test.to(device)
            batch_masks_test = batch_masks_test.to(device)

            batch_masks_test = batch_masks_test.unsqueeze(-1)

            batch_size = batch_windows_test.shape[0]

            h0 = torch.zeros(
                NUM_LAYERS,
                batch_size,
                HIDDEN_SIZE,
                device=device
            )

            reconstruction_test, _ = model(
                batch_windows_test,
                h0
            )

            error_test = (
                batch_windows_test - reconstruction_test
            ) ** 2

            masked_error_test = error_test * batch_masks_test

            window_error = (
                torch.sum(masked_error_test, dim=(1, 2))
                / (
                    torch.sum(batch_masks_test, dim=(1, 2))
                    * batch_windows_test.shape[-1]
                )
            )

            test_errors.extend(window_error.cpu().tolist())

    for i, error in enumerate(test_errors):
        print(f"Window {i}: Reconstruction Error = {error:.6f}")

    return test_errors


# ============================================================
# 19. Create labels
# ============================================================

def create_labels():
    Normal = 0

    cheating_types = {
        1: "Book / Note / Paper",
        2: "Using a Phone"
    }

    labels_first = [0] * 1054
    labels_second = [0] * 1365
    labels_third = [0] * 1060
    labels_forth = [0] * 1161

    label_ranges_first = []
    label_ranges_second = []

    label_ranges_third = [
        (47, 64, 1),
        (246, 258, 1),
        (388, 420, 1),
        (423, 427, 1),
        (476, 524, 2),
        (687, 709, 1)
    ]

    label_ranges_forth = [
        (514, 515, 1),
        (516, 532, 1)
    ]

    window_labels_first = []
    window_labels_second = []
    window_labels_third = []
    window_labels_forth = []

    for start, end, label in label_ranges_first:
        for i in range(start, end + 1):
            labels_first[i] = label

    for i in range(0, len(labels_first), STRIDE):
        window_labels_first.append(labels_first[i:i + WINDOW_SIZE])

    for start, end, label in label_ranges_second:
        for i in range(start, end + 1):
            labels_second[i] = label

    for i in range(0, len(labels_second), STRIDE):
        window_labels_second.append(labels_second[i:i + WINDOW_SIZE])

    for start, end, label in label_ranges_third:
        for i in range(start, end + 1):
            labels_third[i] = label

    for i in range(0, len(labels_third), STRIDE):
        window_labels_third.append(labels_third[i:i + WINDOW_SIZE])

    for start, end, label in label_ranges_forth:
        for i in range(start, end + 1):
            labels_forth[i] = label

    for i in range(0, len(labels_forth), STRIDE):
        window_labels_forth.append(labels_forth[i:i + WINDOW_SIZE])

    window_labels = (
        window_labels_first
        + window_labels_second
        + window_labels_third
        + window_labels_forth
    )

    windows_labels_validation = []
    masks_labels_validation = []

    for current_window in window_labels:
        current_window = torch.tensor(current_window)

        if current_window.shape[0] == WINDOW_SIZE:
            windows_labels_validation.append(current_window)

            mask = torch.ones(WINDOW_SIZE)
            masks_labels_validation.append(mask)

        else:
            padding_length = WINDOW_SIZE - current_window.shape[0]

            padding = torch.zeros(padding_length)

            mask = torch.ones(current_window.shape[0])
            mask_padding = torch.zeros(padding_length)

            ending_window_labels = torch.cat(
                [current_window, padding]
            )

            ending_mask_labels = torch.cat(
                [mask, mask_padding]
            )

            windows_labels_validation.append(ending_window_labels)
            masks_labels_validation.append(ending_mask_labels)

    return (
        Normal,
        cheating_types,
        window_labels,
        windows_labels_validation,
        masks_labels_validation
    )


# ============================================================
# 20. Save test summary table
# ============================================================

def save_test_summary(test_errors):
    test_results = pd.DataFrame({
        "Component": [
            "Modality",
            "Model",
            "Input",
            "Features",
            "Window size",
            "Output",
            "Score level",
            "Test windows",
            "Mean reconstruction error",
            "Median reconstruction error",
            "Standard deviation",
            "Minimum reconstruction error",
            "Maximum reconstruction error",
            "Final decision"
        ],

        "Description": [
            "Mobile",
            "MultiLayerGRU",
            "Mobile feature tensor",
            "Wrist, finger-tip and thumb-tip distances",
            "10 timesteps",
            "Reconstruction error",
            "Window-level anomaly score",
            len(test_errors),
            pd.Series(test_errors).mean(),
            pd.Series(test_errors).median(),
            pd.Series(test_errors).std(),
            pd.Series(test_errors).min(),
            pd.Series(test_errors).max(),
            "Used as input for multimodal fusion"
        ]
    })

    display(test_results)

    test_results.to_csv(
        "mobile_test_summary_table.csv",
        index=False
    )

    return test_results


# ============================================================
# 21. Calculate class-wise validation errors
# ============================================================

def calculate_class_wise_errors(
    timestep_errors_validation,
    windows_labels_validation
):
    errors = timestep_errors_validation.cpu().flatten()

    labels = torch.stack(
        windows_labels_validation
    ).flatten()

    if len(errors) != len(labels):
        raise ValueError(
            f"Mismatch between validation errors ({len(errors)}) "
            f"and labels ({len(labels)})."
        )

    for label, name in [(0, "Normal"), (2, "Cheating")]:
        class_errors = errors[labels == label]

        print(
            name,
            "N =", len(class_errors),
            "Mean =", class_errors.mean().item(),
            "Median =", class_errors.median().item(),
            "Std =", class_errors.std().item(),
            "Min =", class_errors.min().item(),
            "Max =", class_errors.max().item()
        )


# ============================================================
# 22. Save validation summary table
# ============================================================

def save_validation_summary(
    timestep_errors_validation,
    windows_labels_validation
):
    errors = timestep_errors_validation.cpu().flatten()

    labels = torch.stack(
        windows_labels_validation
    ).flatten()

    if len(errors) != len(labels):
        raise ValueError(
            f"Mismatch between validation errors ({len(errors)}) "
            f"and labels ({len(labels)})."
        )

    normal_errors = errors[labels == 0]
    cheating_errors = errors[labels == 2]

    mobile_summary = {
        "Component": [
            "Modality",
            "Model",
            "Input",
            "Features",
            "Window size",
            "Output",
            "Score level",
            "Normal timesteps",
            "Cheating timesteps",
            "Normal mean error",
            "Cheating mean error",
            "Final decision"
        ],

        "Description": [
            "Mobile",
            "MultiLayerGRU",
            "Mobile feature tensor",
            "Wrist, finger-tip and thumb-tip distances",
            "10 timesteps",
            "Reconstruction error",
            "Timestep-level anomaly score",
            len(normal_errors),
            len(cheating_errors),
            (
                normal_errors.mean().item()
                if len(normal_errors) > 0
                else float("nan")
            ),
            (
                cheating_errors.mean().item()
                if len(cheating_errors) > 0
                else float("nan")
            ),
            "Used as input for multimodal fusion"
        ]
    }

    mobile_table = pd.DataFrame(mobile_summary)

    display(mobile_table)

    mobile_table.to_csv(
        "mobile_validation_summary_table.csv",
        index=False
    )

    return mobile_table


# ============================================================
# 23. Main pipeline
# ============================================================

def main():
    setup_environment()

    # Load data
    paths, datas = load_pkl_files()

    # Feature extraction
    all_features = extract_features(datas)

    # Group features
    grouped_features = group_features(all_features)

    # Build video representations
    all_video_representations = build_video_representations(
        all_features,
        grouped_features
    )

    # Split train, validation and test videos
    (
        all_video_representations,
        all_video_representations_validation,
        all_video_representations_test
    ) = split_videos(all_video_representations)

    # Create windows
    (
        windows_validation,
        masks_validation,
        video_index
    ) = create_validation_windows(
        all_video_representations_validation
    )

    windows_test, masks_test = create_test_windows(
        all_video_representations_test
    )

    windows, masks = create_training_windows(
        all_video_representations
    )

    # Normalize
    (
        normalized_features,
        normalized_validation_features,
        normalized_test_features
    ) = normalize_features(
        windows,
        windows_validation,
        windows_test
    )

    # DataLoaders
    (
        train_loader,
        validatation_loader,
        test_loader
    ) = create_data_loaders(
        normalized_features,
        normalized_validation_features,
        normalized_test_features,
        masks,
        masks_validation,
        masks_test
    )

    # Get first training batch to determine input dimension
    batch_windows, batch_masks = next(iter(train_loader))

    batch_windows = batch_windows.to(device)
    batch_masks = batch_masks.to(device)
    batch_masks = batch_masks.unsqueeze(-1)

    input_dim = batch_windows.shape[-1]

    # Build model
    model, optimizer = build_model(input_dim)

    # Train
    (
        epoch_losses,
        batch_windows,
        batch_masks,
        h0
    ) = train_model(
        model,
        optimizer,
        train_loader,
        input_dim
    )

    # Batch diagnostics
    (
        loss,
        loss_validation,
        loss_test
    ) = run_batch_diagnostics(
        model,
        batch_windows,
        batch_masks,
        h0,
        validatation_loader,
        test_loader
    )

    # Validation errors
    (
        validation_errors,
        timestep_errors_validation
    ) = calculate_validation_errors(
        model,
        validatation_loader
    )

    # Test errors
    test_errors = calculate_test_errors(
        model,
        test_loader
    )

    # Labels
    (
        Normal,
        cheating_types,
        window_labels,
        windows_labels_validation,
        masks_labels_validation
    ) = create_labels()

    # Test report
    test_results = save_test_summary(test_errors)

    # Class-wise validation errors
    calculate_class_wise_errors(
        timestep_errors_validation,
        windows_labels_validation
    )

    # Validation report
    mobile_table = save_validation_summary(
        timestep_errors_validation,
        windows_labels_validation
    )


# ============================================================
# 24. Run pipeline
# ============================================================

if __name__ == "__main__":
    main()
