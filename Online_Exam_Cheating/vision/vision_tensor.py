
# ============================================================
# 1. Imports
# ============================================================

import os
import pickle
import argparse

import torch
import pandas as pd

from models.LSTMAE import LSTMAE


# ============================================================
# 2. Load .pkl files
# ============================================================

DATA_DIR = "/content/drive/MyDrive/Online_Exam_Cheating"

paths = [
    path
    for path in os.listdir(DATA_DIR)
    if path.endswith(".pkl")
]

# مهم: ترتیب فایل‌ها ثابت باشد
paths = sorted(paths)

datas = []

for path in paths:

    with open(
        os.path.join(DATA_DIR, path),
        "rb"
    ) as f:

        datas.append(pickle.load(f))


print("Number of loaded files:", len(datas))


# ============================================================
# 3. Build frame-level vision features
# ============================================================

def build_vision_features(datas):

    all_features = []

    for data_index in range(len(datas)):

        video = datas[data_index][0]

        # ----------------------------------------------------
        # Feature containers
        # ----------------------------------------------------

        gaze_face_confidence = []
        gaze_pitch = []
        gaze_yaw = []

        left_iris_x = []
        left_iris_y = []
        right_iris_x = []
        right_iris_y = []

        iris_valid = []

        face_count = []

        head_pose_pitch = []
        head_pose_yaw = []
        head_pose_roll = []

        pose_features_left_elbow_angle = []
        pose_features_right_elbow_angle = []

        pose_features_left_wrist_distance = []
        pose_features_right_wrist_distance = []

        pose_features_left_elbow_distance = []
        pose_features_right_elbow_distance = []

        object_confidence = []
        object_class = []

        features = {
            "video_id": video["video_id"],
            "tensor": []
        }

        # ====================================================
        # Process frames
        # ====================================================

        for j in range(len(video["frames"])):

            frame = video["frames"][j]

            # =================================================
            # Objects
            # =================================================

            objects = frame["objects"]

            if objects is None or len(objects) == 0:

                object_confidence.append(0)
                object_class.append(3)

            else:

                best_object = max(
                    objects,
                    key=lambda obj: obj["conf"]
                )

                object_confidence.append(
                    best_object["conf"]
                )

                object_class.append(
                    best_object["class"]
                )

            # =================================================
            # Face
            # =================================================

            face = frame["face"]

            if face is None:

                face_count.append(0)

            else:

                if face["face_count"] is None:
                    face_count.append(0)
                else:
                    face_count.append(
                        face["face_count"]
                    )

            # =================================================
            # Head pose
            # =================================================

            head_pose = frame["head_pose"]

            if head_pose is None:

                head_pose_pitch.append(0)
                head_pose_yaw.append(0)
                head_pose_roll.append(0)

            else:

                head_pose_pitch.append(
                    0 if head_pose["pitch"] is None
                    else head_pose["pitch"]
                )

                head_pose_yaw.append(
                    0 if head_pose["yaw"] is None
                    else head_pose["yaw"]
                )

                head_pose_roll.append(
                    0 if head_pose["roll"] is None
                    else head_pose["roll"]
                )

            # =================================================
            # Pose
            # =================================================

            pose = frame["pose_features"]

            if pose is None:

                pose_features_left_elbow_angle.append(0)
                pose_features_right_elbow_angle.append(0)

                pose_features_left_wrist_distance.append(0)
                pose_features_right_wrist_distance.append(0)

                pose_features_left_elbow_distance.append(0)
                pose_features_right_elbow_distance.append(0)

            else:

                pose_features_left_elbow_angle.append(
                    0 if pose["left_elbow_angle"] is None
                    else pose["left_elbow_angle"]
                )

                pose_features_right_elbow_angle.append(
                    0 if pose["right_elbow_angle"] is None
                    else pose["right_elbow_angle"]
                )

                pose_features_left_wrist_distance.append(
                    0 if pose["left_wrist_distance"] is None
                    else pose["left_wrist_distance"]
                )

                pose_features_right_wrist_distance.append(
                    0 if pose["right_wrist_distance"] is None
                    else pose["right_wrist_distance"]
                )

                pose_features_left_elbow_distance.append(
                    0 if pose["left_elbow_distance"] is None
                    else pose["left_elbow_distance"]
                )

                pose_features_right_elbow_distance.append(
                    0 if pose["right_elbow_distance"] is None
                    else pose["right_elbow_distance"]
                )

            # =================================================
            # Gaze
            # =================================================

            gaze = frame["gaze"]

            if gaze is None:

                gaze_pitch.append(0)
                gaze_yaw.append(0)
                gaze_face_confidence.append(0)

                left_iris_x.append(0)
                left_iris_y.append(0)

                right_iris_x.append(0)
                right_iris_y.append(0)

                iris_valid.append(0)

            else:

                gaze_pitch.append(
                    0 if gaze["pitch"] is None
                    else gaze["pitch"]
                )

                gaze_yaw.append(
                    0 if gaze["yaw"] is None
                    else gaze["yaw"]
                )

                gaze_face_confidence.append(
                    0 if gaze["face_confidence"] is None
                    else gaze["face_confidence"]
                )

                left_iris_x.append(
                    0 if gaze["left_iris_x"] is None
                    else gaze["left_iris_x"]
                )

                left_iris_y.append(
                    0 if gaze["left_iris_y"] is None
                    else gaze["left_iris_y"]
                )

                right_iris_x.append(
                    0 if gaze["right_iris_x"] is None
                    else gaze["right_iris_x"]
                )

                right_iris_y.append(
                    0 if gaze["right_iris_y"] is None
                    else gaze["right_iris_y"]
                )

                iris_valid.append(
                    0 if gaze["iris_valid"] is None
                    else gaze["iris_valid"]
                )

        # ====================================================
        # Convert to tensors
        # ====================================================

        gaze_pitch = torch.tensor(gaze_pitch)
        gaze_yaw = torch.tensor(gaze_yaw)
        gaze_face_confidence = torch.tensor(
            gaze_face_confidence
        )

        left_iris_x = torch.tensor(left_iris_x)
        left_iris_y = torch.tensor(left_iris_y)

        right_iris_x = torch.tensor(right_iris_x)
        right_iris_y = torch.tensor(right_iris_y)

        iris_valid = torch.tensor(iris_valid)

        face_count = torch.tensor(face_count)

        head_pose_pitch = torch.tensor(
            head_pose_pitch
        )

        head_pose_yaw = torch.tensor(
            head_pose_yaw
        )

        head_pose_roll = torch.tensor(
            head_pose_roll
        )

        pose_features_left_elbow_angle = torch.tensor(
            pose_features_left_elbow_angle
        )

        pose_features_right_elbow_angle = torch.tensor(
            pose_features_right_elbow_angle
        )

        pose_features_left_wrist_distance = torch.tensor(
            pose_features_left_wrist_distance
        )

        pose_features_right_wrist_distance = torch.tensor(
            pose_features_right_wrist_distance
        )

        pose_features_left_elbow_distance = torch.tensor(
            pose_features_left_elbow_distance
        )

        pose_features_right_elbow_distance = torch.tensor(
            pose_features_right_elbow_distance
        )

        object_confidence = torch.tensor(
            object_confidence
        )

        object_class = torch.tensor(
            object_class
        )

        # ====================================================
        # Store features
        # ====================================================

        features["tensor"].append(gaze_pitch)
        features["tensor"].append(gaze_yaw)
        features["tensor"].append(gaze_face_confidence)

        features["tensor"].append(left_iris_x)
        features["tensor"].append(left_iris_y)

        features["tensor"].append(right_iris_x)
        features["tensor"].append(right_iris_y)

        features["tensor"].append(iris_valid)

        features["tensor"].append(face_count)

        features["tensor"].append(head_pose_pitch)
        features["tensor"].append(head_pose_yaw)
        features["tensor"].append(head_pose_roll)

        features["tensor"].append(
            pose_features_left_elbow_angle
        )

        features["tensor"].append(
            pose_features_right_elbow_angle
        )

        features["tensor"].append(
            pose_features_left_wrist_distance
        )

        features["tensor"].append(
            pose_features_right_wrist_distance
        )

        features["tensor"].append(
            pose_features_left_elbow_distance
        )

        features["tensor"].append(
            pose_features_right_elbow_distance
        )

        features["tensor"].append(object_confidence)
        features["tensor"].append(object_class)

        all_features.append(features)

    return all_features


all_features = build_vision_features(datas)


# ============================================================
# 4. Group features by modality
# ============================================================

def group_features(all_features):

    grouped_features = []

    for i in range(len(all_features)):

        gaze = torch.stack(
            all_features[i]["tensor"][0:8],
            dim=1
        )

        face = torch.stack(
            all_features[i]["tensor"][8:9],
            dim=1
        )

        head_pose = torch.stack(
            all_features[i]["tensor"][9:12],
            dim=1
        )

        pose = torch.stack(
            all_features[i]["tensor"][12:18],
            dim=1
        )

        objects = torch.stack(
            all_features[i]["tensor"][18:20],
            dim=1
        )

        grouped_features.append({
            "video_id": all_features[i]["video_id"],
            "gaze": gaze,
            "face": face,
            "head_pose": head_pose,
            "pose": pose,
            "objects": objects
        })

    return grouped_features


grouped_features = group_features(
    all_features
)


# ============================================================
# 5. Create temporal video representations
# ============================================================

def create_video_representations(
    all_features,
    grouped_features
):

    all_video_representations = []

    for i in range(len(all_features)):

        num_frames = int(
            all_features[i]["tensor"][0].shape[0]
        )

        video_representations = []

        for t_index in range(
            0,
            num_frames,
            25
        ):

            time_start = t_index
            time_end = min(
                t_index + 25,
                num_frames
            )

            frames_in_timestep = (
                time_end - time_start
            )

            # =================================================
            # Extract temporal slices
            # =================================================

            gaze = grouped_features[i]["gaze"][
                time_start:time_end
            ]

            face = grouped_features[i]["face"][
                time_start:time_end
            ]

            head_pose = grouped_features[i]["head_pose"][
                time_start:time_end
            ]

            pose = grouped_features[i]["pose"][
                time_start:time_end
            ]

            objects = grouped_features[i]["objects"][
                time_start:time_end
            ]

            # =================================================
            # Gaze
            # =================================================

            gaze_face_confidence_observation = gaze[0, 2]
            gaze_pitch_observation = gaze[0, 1]
            gaze_yaw_observation = gaze[0, 0]

            left_iris_x_observation = gaze[0, 3]
            left_iris_y_observation = gaze[0, 4]

            right_iris_x_observation = gaze[0, 5]
            right_iris_y_observation = gaze[0, 6]

            iris_valid_observation = gaze[0, 7]

            # =================================================
            # Face
            # =================================================

            face_count_observation = face[0, 0]

            # =================================================
            # Head pose
            # =================================================

            head_pose_pitch_observation = torch.median(
                head_pose[:, 0]
            )

            head_pose_yaw_observation = torch.median(
                head_pose[:, 1]
            )

            head_pose_roll_observation = torch.median(
                head_pose[:, 2]
            )

            if frames_in_timestep > 1:

                head_pitch_changes = (
                    head_pose[1:, 0]
                    - head_pose[:-1, 0]
                )

                head_yaw_changes = (
                    head_pose[1:, 1]
                    - head_pose[:-1, 1]
                )

                head_roll_changes = (
                    head_pose[1:, 2]
                    - head_pose[:-1, 2]
                )

                head_pose_pitch_observation_mean = (
                    torch.mean(
                        torch.abs(
                            head_pitch_changes
                        )
                    )
                )

                head_pose_yaw_observation_mean = (
                    torch.mean(
                        torch.abs(
                            head_yaw_changes
                        )
                    )
                )

                head_pose_roll_observation_mean = (
                    torch.mean(
                        torch.abs(
                            head_roll_changes
                        )
                    )
                )

            else:

                head_pose_pitch_observation_mean = (
                    torch.tensor(0.0)
                )

                head_pose_yaw_observation_mean = (
                    torch.tensor(0.0)
                )

                head_pose_roll_observation_mean = (
                    torch.tensor(0.0)
                )

            # =================================================
            # Pose
            # =================================================

            pose_1_features_left_elbow_angle_observation = (
                pose[0, 0]
            )

            pose_1_features_right_elbow_angle_observation = (
                pose[0, 1]
            )

            pose_1_features_left_wrist_distance_observation = (
                pose[0, 2]
            )

            pose_1_features_right_wrist_distance_observation = (
                pose[0, 3]
            )

            pose_1_features_left_elbow_distance_observation = (
                pose[0, 4]
            )

            pose_1_features_right_elbow_distance_observation = (
                pose[0, 5]
            )

            # -------------------------------------------------
            # Second and third pose observations
            # -------------------------------------------------

            if frames_in_timestep >= 20:

                pose_2_features_left_elbow_angle_observation = (
                    pose[10, 0]
                )

                pose_2_features_right_elbow_angle_observation = (
                    pose[10, 1]
                )

                pose_2_features_left_wrist_distance_observation = (
                    pose[10, 2]
                )

                pose_2_features_right_wrist_distance_observation = (
                    pose[10, 3]
                )

                pose_2_features_left_elbow_distance_observation = (
                    pose[10, 4]
                )

                pose_2_features_right_elbow_distance_observation = (
                    pose[10, 5]
                )

            elif frames_in_timestep >= 10:

                pose_2_features_left_elbow_angle_observation = torch.tensor(0.0)

                pose_2_features_right_elbow_angle_observation = torch.tensor(0.0)

                pose_2_features_left_wrist_distance_observation = torch.tensor(0.0)

                pose_2_features_right_wrist_distance_observation = torch.tensor(0.0)

                pose_2_features_left_elbow_distance_observation = torch.tensor(0.0)

                pose_2_features_right_elbow_distance_observation = torch.tensor(0.0)

            else:

                pose_2_features_left_elbow_angle_observation = torch.tensor(0.0)

                pose_2_features_right_elbow_angle_observation = torch.tensor(0.0)

                pose_2_features_left_wrist_distance_observation = torch.tensor(0.0)

                pose_2_features_right_wrist_distance_observation = torch.tensor(0.0)

                pose_2_features_left_elbow_distance_observation = torch.tensor(0.0)

                pose_2_features_right_elbow_distance_observation = torch.tensor(0.0)

            # =================================================
            # Objects
            # =================================================

            object_confidence_observation = objects[0, 0]
            object_class_observation = objects[0, 1]

            # =================================================
            # Final 29-dimensional representation
            # =================================================

            T = torch.stack([

                gaze_face_confidence_observation,
                gaze_pitch_observation,
                gaze_yaw_observation,

                left_iris_x_observation,
                left_iris_y_observation,

                right_iris_x_observation,
                right_iris_y_observation,

                iris_valid_observation,

                face_count_observation,

                head_pose_pitch_observation,
                head_pose_yaw_observation,
                head_pose_roll_observation,

                head_pose_pitch_observation_mean,
                head_pose_yaw_observation_mean,
                head_pose_roll_observation_mean,

                pose_1_features_left_elbow_angle_observation,
                pose_2_features_left_elbow_angle_observation,

                pose_1_features_right_elbow_angle_observation,
                pose_2_features_right_elbow_angle_observation,

                pose_1_features_left_wrist_distance_observation,
                pose_2_features_left_wrist_distance_observation,

                pose_1_features_right_wrist_distance_observation,
                pose_2_features_right_wrist_distance_observation,

                pose_1_features_left_elbow_distance_observation,
                pose_2_features_left_elbow_distance_observation,

                pose_1_features_right_elbow_distance_observation,
                pose_2_features_right_elbow_distance_observation,

                object_confidence_observation,
                object_class_observation
            ])

            video_representations.append(T)

        video_representations = torch.stack(
            video_representations
        )

        all_video_representations.append(
            video_representations
        )

    return all_video_representations


all_video_representations = create_video_representations(
    all_features,
    grouped_features
)


print(
    "First video representation shape:",
    all_video_representations[0].shape
)


# ============================================================
# 6. Train / Validation / Test split
# ============================================================

all_video_representations_validation = (
    all_video_representations[0:4]
)

all_video_representations_test = (
    all_video_representations[4:8]
)

all_video_representations = (
    all_video_representations[8:]
)


# ============================================================
# 7. Create windows + masks
# ============================================================

def create_windows(
    video_representations,
    window_size=10,
    stride=8
):

    windows = []
    masks = []
    video_index = []

    for i, video in enumerate(
        video_representations
    ):

        for j in range(
            0,
            video.shape[0],
            stride
        ):

            current_window = video[
                j:j + window_size
            ]

            # ------------------------------------------------
            # Full window
            # ------------------------------------------------

            if current_window.shape[0] == window_size:

                windows.append(
                    current_window
                )

                mask = torch.ones(
                    window_size
                )

                masks.append(mask)

            # ------------------------------------------------
            # Final incomplete window
            # ------------------------------------------------

            elif (
                current_window.shape[0] > 0
                and current_window.shape[0] < window_size
            ):

                padding_length = (
                    window_size
                    - current_window.shape[0]
                )

                padding = torch.zeros(
                    padding_length,
                    video.shape[1]
                )

                ending_window = torch.cat(
                    [
                        current_window,
                        padding
                    ],
                    dim=0
                )

                mask = torch.ones(
                    current_window.shape[0]
                )

                mask_padding = torch.zeros(
                    padding_length
                )

                ending_mask = torch.cat(
                    [
                        mask,
                        mask_padding
                    ],
                    dim=0
                )

                windows.append(
                    ending_window
                )

                masks.append(
                    ending_mask
                )

                # Store location of padding
                for k in range(
                    len(ending_mask)
                ):

                    if ending_mask[k] == 0:

                        video_index.append(
                            (
                                len(windows) - 1,
                                i,
                                k + j
                            )
                        )

                        break

    return (
        windows,
        masks,
        video_index
    )


windows, masks, video_index = create_windows(
    all_video_representations,
    window_size=10,
    stride=8
)

windows_validation, masks_validation, video_index_validation = (
    create_windows(
        all_video_representations_validation,
        window_size=10,
        stride=8
    )
)

windows_test, masks_test, video_index_test = create_windows(
    all_video_representations_test,
    window_size=10,
    stride=8
)


print("Train windows:", len(windows))
print("Validation windows:", len(windows_validation))
print("Test windows:", len(windows_test))


# ============================================================
# 8. Convert windows to tensors
# ============================================================

train_tensor = torch.stack(
    windows
)

validation_tensor = torch.stack(
    windows_validation
)

test_tensor = torch.stack(
    windows_test
)


print("Train tensor:", train_tensor.shape)
print("Validation tensor:", validation_tensor.shape)
print("Test tensor:", test_tensor.shape)


# ============================================================
# 9. Normalize features
#    IMPORTANT: statistics are calculated only on TRAIN
# ============================================================

def normalize_features(
    train_tensor,
    validation_tensor,
    test_tensor
):

    feature_max = train_tensor.amax(
        dim=(0, 1)
    )

    feature_min = train_tensor.amin(
        dim=(0, 1)
    )

    normalized_features = []
    normalized_validation_features = []
    normalized_test_features = []

    for i in range(
        train_tensor.shape[2]
    ):

        # Remove:
        # feature 8  -> face_count
        # feature 28 -> object_class

        if i == 8 or i == 28:
            continue

        denominator = (
            feature_max[i]
            - feature_min[i]
        )

        if denominator == 0:

            z = torch.zeros_like(
                train_tensor[:, :, i]
            )

            z_validation = torch.zeros_like(
                validation_tensor[:, :, i]
            )

            z_test = torch.zeros_like(
                test_tensor[:, :, i]
            )

        else:

            z = (
                train_tensor[:, :, i]
                - feature_min[i]
            ) / denominator

            z_validation = (
                validation_tensor[:, :, i]
                - feature_min[i]
            ) / denominator

            z_test = (
                test_tensor[:, :, i]
                - feature_min[i]
            ) / denominator

        normalized_features.append(z)
        normalized_validation_features.append(
            z_validation
        )
        normalized_test_features.append(
            z_test
        )

    normalized_features = torch.stack(
        normalized_features
    ).permute(1, 2, 0)

    normalized_validation_features = torch.stack(
        normalized_validation_features
    ).permute(1, 2, 0)

    normalized_test_features = torch.stack(
        normalized_test_features
    ).permute(1, 2, 0)

    return (
        normalized_features,
        normalized_validation_features,
        normalized_test_features,
        feature_min,
        feature_max
    )


(
    normalized_features,
    normalized_validation_features,
    normalized_test_features,
    feature_min,
    feature_max
) = normalize_features(
    train_tensor,
    validation_tensor,
    test_tensor
)


print(
    "Normalized train shape:",
    normalized_features.shape
)

print(
    "Normalized validation shape:",
    normalized_validation_features.shape
)

print(
    "Normalized test shape:",
    normalized_test_features.shape
)

print(
    "Train min:",
    normalized_features.min().item()
)

print(
    "Train max:",
    normalized_features.max().item()
)


# ============================================================
# 10. Dataset
# ============================================================

class SequenceDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        windows,
        masks
    ):

        self.windows = windows
        self.masks = masks

    def __len__(self):

        return len(
            self.windows
        )

    def __getitem__(
        self,
        idx
    ):

        return (
            self.windows[idx],
            self.masks[idx]
        )


# ============================================================
# 11. DataLoaders
# ============================================================

train_dataset = SequenceDataset(
    normalized_features,
    masks
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


validation_dataset = SequenceDataset(
    normalized_validation_features,
    masks_validation
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)


test_dataset = SequenceDataset(
    normalized_test_features,
    masks_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# 12. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 13. LSTM Autoencoder
# ============================================================

lstmae = LSTMAE(
    input_size=27,
    hidden_size=256,
    dropout_ratio=0.0,
    seq_len=10
)

lstmae.to(device)


# ============================================================
# 14. Train model
# ============================================================

def train_model(
    model,
    train_loader,
    device,
    epochs=100,
    lr=0.001
):

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )

    epoch_losses = []

    for epoch in range(epochs):

        model.train()

        batch_losses = []

        for (
            batch_windows,
            batch_masks
        ) in train_loader:

            batch_windows = batch_windows.to(
                device
            )

            batch_masks = batch_masks.to(
                device
            )

            batch_masks = batch_masks.unsqueeze(
                -1
            )

            # --------------------------------------------
            # Reconstruction
            # --------------------------------------------

            reconstruction = model(
                batch_windows
            )

            # --------------------------------------------
            # Squared reconstruction error
            # --------------------------------------------

            error = (
                batch_windows
                - reconstruction
            ) ** 2

            # --------------------------------------------
            # Ignore padded timesteps
            # --------------------------------------------

            masked_error = (
                error
                * batch_masks
            )

            # --------------------------------------------
            # Mean reconstruction error
            # over valid timestep × 27 features
            # --------------------------------------------

            loss = (
                torch.sum(masked_error)
                /
                (
                    torch.sum(batch_masks)
                    * 27
                )
            )

            # --------------------------------------------
            # Backpropagation
            # --------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_losses.append(
                loss.item()
            )

        epoch_loss = (
            sum(batch_losses)
            / len(batch_losses)
        )

        epoch_losses.append(
            epoch_loss
        )

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {epoch_loss:.6f}"
        )

    return (
        model,
        optimizer,
        epoch_losses
    )


lstmae, optimizer, epoch_losses = train_model(
    lstmae,
    train_loader,
    device,
    epochs=100,
    lr=0.001
)


# ============================================================
# 15. Validation reconstruction errors
# ============================================================

def calculate_validation_errors(
    model,
    validation_loader,
    device
):

    model.eval()

    validation_errors = []
    timestep_errors = []
    batch_losses = []

    with torch.no_grad():

        for (
            batch_windows,
            batch_masks
        ) in validation_loader:

            batch_windows = batch_windows.to(
                device
            )

            batch_masks = batch_masks.to(
                device
            )

            batch_masks = batch_masks.unsqueeze(
                -1
            )

            reconstruction = model(
                batch_windows
            )

            error = (
                batch_windows
                - reconstruction
            ) ** 2

            masked_error = (
                error
                * batch_masks
            )

            # --------------------------------------------
            # Timestep-level anomaly score
            # --------------------------------------------

            timestep_error = (
                masked_error.mean(
                    dim=2
                )
            )

            timestep_errors.append(
                timestep_error
            )

            # --------------------------------------------
            # Window-level reconstruction error
            # --------------------------------------------

            window_error = (
                torch.sum(
                    masked_error,
                    dim=(1, 2)
                )
                /
                (
                    torch.sum(
                        batch_masks,
                        dim=(1, 2)
                    )
                    * 27
                )
            )

            # --------------------------------------------
            # Batch loss
            # --------------------------------------------

            loss = (
                torch.sum(masked_error)
                /
                (
                    torch.sum(batch_masks)
                    * 27
                )
            )

            batch_losses.append(
                loss.item()
            )

            validation_errors.extend(
                window_error.cpu().tolist()
            )

    timestep_errors = torch.cat(
        timestep_errors,
        dim=0
    )

    return (
        batch_losses,
        validation_errors,
        timestep_errors
    )


(
    validation_losses,
    validation_errors,
    timestep_errors_validation
) = calculate_validation_errors(
    lstmae,
    validation_loader,
    device
)


print(
    "Validation timestep errors shape:",
    timestep_errors_validation.shape
)


# ============================================================
# 16. Test reconstruction errors
# ============================================================

def calculate_test_errors(
    model,
    test_loader,
    device
):

    model.eval()

    test_errors = []

    with torch.no_grad():

        for (
            batch_windows,
            batch_masks
        ) in test_loader:

            batch_windows = batch_windows.to(
                device
            )

            batch_masks = batch_masks.to(
                device
            )

            batch_masks = batch_masks.unsqueeze(
                -1
            )

            reconstruction = model(
                batch_windows
            )

            error = (
                batch_windows
                - reconstruction
            ) ** 2

            masked_error = (
                error
                * batch_masks
            )

            window_error = (
                torch.sum(
                    masked_error,
                    dim=(1, 2)
                )
                /
                (
                    torch.sum(
                        batch_masks,
                        dim=(1, 2)
                    )
                    * 27
                )
            )

            test_errors.extend(
                window_error.cpu().tolist()
            )

    return test_errors


test_errors = calculate_test_errors(
    lstmae,
    test_loader,
    device
)


for i, error in enumerate(
    test_errors
):

    print(
        f"Window {i}: "
        f"Reconstruction Error = "
        f"{error:.6f}"
    )


# ============================================================
# 17. Labels
# ============================================================

Normal = 0

cheating_types = {
    1: "Talking to a Person",
    2: "Asking a Friend a Question Over the Phone",
    3: "Using a Phone"
}


# ============================================================
# Label ranges
# ============================================================

label_ranges_first = [
    (1, 5, 1),
    (8, 10, 1),
    (17, 21, 1),
    (170, 179, 1),
    (265, 274, 1),
    (785, 821, 1),
    (823, 829, 1),
    (841, 854, 1),
    (856, 862, 1),
    (875, 877, 1),
    (880, 881, 1),
    (1034, 1035, 1),
    (1039, 1044, 1),
    (1049, 1054, 1),
]

label_ranges_second = [
    (2, 5, 1),
    (14, 23, 1),
    (393, 395, 1),
    (544, 550, 1),
    (554, 566, 1),
    (569, 583, 1),
    (595, 614, 1),
    (618, 620, 1),
    (627, 633, 1),
    (985, 1022, 1),
    (1033, 1042, 1),
    (1053, 1070, 1),
    (1080, 1081, 1),
    (1359, 1362, 1),
]

label_ranges_third = [
    (476, 524, 2),
]

label_ranges_forth = [
    (1, 8, 1),
    (14, 19, 1),
    (124, 134, 1),
    (290, 292, 1),
    (295, 312, 1),
    (329, 333, 1),
    (392, 396, 1),
    (499, 540, 1),
    (810, 857, 1),
    (980, 992, 1),
    (1139, 1149, 1),
    (1152, 1161, 1),
]


# ============================================================
# 18. Create frame-level + window-level labels
# ============================================================

def create_window_labels(
    num_frames,
    label_ranges,
    window_size=10,
    stride=8
):

    labels = [0] * num_frames

    for start, end, label in label_ranges:

        for i in range(
            start,
            end + 1
        ):

            labels[i] = label

    window_labels = []

    for i in range(
        0,
        num_frames,
        stride
    ):

        window_labels.append(
            labels[
                i:i + window_size
            ]
        )

    return (
        labels,
        window_labels
    )


labels_first, window_labels_first = (
    create_window_labels(
        1055,
        label_ranges_first
    )
)

labels_second, window_labels_second = (
    create_window_labels(
        1366,
        label_ranges_second
    )
)

labels_third, window_labels_third = (
    create_window_labels(
        1061,
        label_ranges_third
    )
)

labels_forth, window_labels_forth = (
    create_window_labels(
        1162,
        label_ranges_forth
    )
)


# ============================================================
# 19. Combine validation labels
# ============================================================

window_labels = (
    window_labels_first
    + window_labels_second
    + window_labels_third
    + window_labels_forth
)


# ============================================================
# 20. Prepare padded labels + masks
# ============================================================

def prepare_window_labels(
    window_labels,
    window_size=10
):

    windows_labels = []
    masks_labels = []

    for current_window in window_labels:

        current_window = torch.tensor(
            current_window,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Full window
        # ----------------------------------------------------

        if current_window.shape[0] == window_size:

            windows_labels.append(
                current_window
            )

            masks_labels.append(
                torch.ones(window_size)
            )

        # ----------------------------------------------------
        # Incomplete window
        # ----------------------------------------------------

        else:

            padding_length = (
                window_size
                - current_window.shape[0]
            )

            padding = torch.zeros(
                padding_length,
                dtype=torch.long
            )

            mask = torch.ones(
                current_window.shape[0]
            )

            mask_padding = torch.zeros(
                padding_length
            )

            ending_window_labels = torch.cat(
                [
                    current_window,
                    padding
                ]
            )

            ending_mask_labels = torch.cat(
                [
                    mask,
                    mask_padding
                ]
            )

            windows_labels.append(
                ending_window_labels
            )

            masks_labels.append(
                ending_mask_labels
            )

    return (
        windows_labels,
        masks_labels
    )


(
    windows_labels_validation,
    masks_labels_validation
) = prepare_window_labels(
    window_labels,
    window_size=10
)


# ============================================================
# 21. Calculate class statistics
# ============================================================

def calculate_class_statistics(
    timestep_errors,
    window_labels,
    label_mask=None
):

    errors = (
        timestep_errors
        .cpu()
    )

    labels = (
        torch.stack(window_labels)
    )

    # --------------------------------------------------------
    # Flatten timestep-level errors and labels
    # --------------------------------------------------------

    errors = errors.flatten()
    labels = labels.flatten()

    # --------------------------------------------------------
    # Remove padded timesteps
    # --------------------------------------------------------

    if label_mask is not None:

        label_mask = (
            torch.stack(label_mask)
            .flatten()
            .bool()
        )

        errors = errors[label_mask]
        labels = labels[label_mask]

    # --------------------------------------------------------
    # Normal vs Cheating
    # --------------------------------------------------------

    for label, name in [
        (0, "Normal"),
        (2, "Cheating")
    ]:

        class_errors = (
            errors[
                labels == label
            ]
        )


calculate_class_statistics(
    timestep_errors_validation,
    windows_labels_validation,
    masks_labels_validation
)
