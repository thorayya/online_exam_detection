"""Mobile-modality reconstruction pipeline for online exam-cheating detection.

Expected input records:
    Each .pkl file contains a list of timestep/frame dictionaries. Each record
    has a `video` key and an `objects` key. `objects` is None or a list of
    detected-object dictionaries containing hand-distance features, confidence,
    class_id, and name.

Run from the repository root, for example:
    python mobile_pipeline.py --data-dir mobile --output-dir outputs/mobile

The script imports MultiLayerGRU from models/gru.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

FEATURE_NAMES = [
    "distance_wrist_left",
    "distance_wrist_right",
    "distance_finger_tip_left",
    "distance_finger_tip_right",
    "distance_thumb_tip_left",
    "distance_thumb_tip_right",
    "conf",
    "class_id",
]
MODEL_FEATURE_INDICES = list(range(7))  # class_id (index 7) is excluded.
WINDOW_SIZE = 10
STRIDE = 8
BATCH_SIZE = 32
EPOCHS = 100
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 0.001
SEED = 42

# These are the indices used by the original notebook. Their meaning depends
# on the file order, so the script prints the sorted index-to-file mapping.
VALIDATION_INDICES = [2, 3, 22, 6]
TEST_INDICES = [4, 5, 7, 8]

# Preserve the original split exactly: train = [0:2] + [9:22] + [23:].
VALIDATION_LABEL_LENGTHS = [1054, 1365, 1060, 1161]
VALIDATION_LABEL_RANGES = [
    [],
    [],
    [(47, 64, 1), (246, 258, 1), (388, 420, 1), (423, 427, 1),
     (476, 524, 2), (687, 709, 1)],
    [(514, 515, 1), (516, 532, 1)],
]
LABEL_NAMES = {0: "Normal", 1: "Book / Note / Paper", 2: "Using a Phone"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(data_dir: Path) -> tuple[list[str], list[list[dict[str, Any]]]]:
    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in: {data_dir}")

    records = []
    print("Sorted PKL index -> filename (verify these match the notebook split):")
    for index, path in enumerate(files):
        print(f"  {index}: {path.name}")
        with path.open("rb") as handle:
            records.append(pickle.load(handle))
    return [p.name for p in files], records


def safe_value(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def extract_video_features(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("Encountered an empty video record.")

    values = {name: [] for name in FEATURE_NAMES}
    names: list[str] = []

    for frame_index, frame in enumerate(frames):
        objects = frame.get("objects")
        if not objects:
            for key in FEATURE_NAMES[:6]:
                values[key].append(0.0)
            values["conf"].append(0.0)
            values["class_id"].append(7)
            names.append("unknown")
            continue

        # A missing confidence is treated as zero for robust selection.
        best_object = max(objects, key=lambda obj: safe_value(obj.get("conf"), 0.0))
        for key in FEATURE_NAMES[:6]:
            values[key].append(float(safe_value(best_object.get(key), 0.0)))
        values["conf"].append(float(safe_value(best_object.get("conf"), 0.0)))
        values["class_id"].append(int(safe_value(best_object.get("class_id"), 7)))
        names.append(str(safe_value(best_object.get("name"), "unknown")))

    tensor = torch.tensor(
        np.column_stack([values[name] for name in FEATURE_NAMES]),
        dtype=torch.float32,
    )
    return {
        "video_id": frames[0].get("video", "unknown_video"),
        "tensor": tensor,  # [T, 8]
        "name": names,
    }


def make_windows(
    videos: list[torch.Tensor], window_size: int = WINDOW_SIZE, stride: int = STRIDE
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    windows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    padded_positions: list[tuple[int, int, int]] = []

    for video_index, video in enumerate(videos):
        if video.ndim != 2 or video.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Video {video_index} has unexpected shape {tuple(video.shape)}")

        for start in range(0, video.shape[0], stride):
            current = video[start : start + window_size]
            if current.shape[0] == 0:
                continue
            real_length = current.shape[0]
            if real_length < window_size:
                padding = torch.zeros(
                    window_size - real_length, video.shape[1], dtype=video.dtype
                )
                current = torch.cat([current, padding], dim=0)
            mask = torch.zeros(window_size, dtype=torch.float32)
            mask[:real_length] = 1.0
            windows.append(current)
            masks.append(mask)
            if real_length < window_size:
                padded_positions.append((len(windows) - 1, video_index, start + real_length))

    if not windows:
        raise ValueError("No windows were created; check the input videos.")
    return torch.stack(windows), torch.stack(masks), padded_positions


def normalize_windows(
    train_windows: torch.Tensor,
    validation_windows: torch.Tensor,
    test_windows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit min/max on training data only; exclude class_id (feature index 7)."""
    feature_max = train_windows.amax(dim=(0, 1))
    feature_min = train_windows.amin(dim=(0, 1))
    denominator = feature_max - feature_min
    # Constant features are mapped to zero rather than dividing by zero.
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

    train = (train_windows[:, :, MODEL_FEATURE_INDICES] - feature_min[MODEL_FEATURE_INDICES]) / denominator[MODEL_FEATURE_INDICES]
    validation = (validation_windows[:, :, MODEL_FEATURE_INDICES] - feature_min[MODEL_FEATURE_INDICES]) / denominator[MODEL_FEATURE_INDICES]
    test = (test_windows[:, :, MODEL_FEATURE_INDICES] - feature_min[MODEL_FEATURE_INDICES]) / denominator[MODEL_FEATURE_INDICES]
    return train, validation, test, feature_min, feature_max


class SequenceDataset(Dataset):
    def __init__(self, windows: torch.Tensor, masks: torch.Tensor):
        if len(windows) != len(masks):
            raise ValueError(f"Window/mask count mismatch: {len(windows)} vs {len(masks)}")
        self.windows = windows
        self.masks = masks

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.masks[index]


def masked_reconstruction_loss(
    inputs: torch.Tensor, reconstruction: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    feature_mask = masks.unsqueeze(-1)
    squared_error = (inputs - reconstruction).pow(2) * feature_mask
    denominator = feature_mask.sum() * inputs.shape[-1]
    return squared_error.sum() / denominator.clamp_min(1.0)


def forward_batch(model: torch.nn.Module, batch_windows: torch.Tensor) -> torch.Tensor:
    batch_size = batch_windows.shape[0]
    h0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=batch_windows.device)
    reconstruction, _ = model(batch_windows, h0)
    return reconstruction


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_windows, batch_masks in train_loader:
            batch_windows = batch_windows.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad()
            reconstruction = forward_batch(model, batch_windows)
            loss = masked_reconstruction_loss(batch_windows, reconstruction, batch_masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch_windows, batch_masks in validation_loader:
                batch_windows = batch_windows.to(device)
                batch_masks = batch_masks.to(device)
                reconstruction = forward_batch(model, batch_windows)
                loss = masked_reconstruction_loss(batch_windows, reconstruction, batch_masks)
                validation_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch + 1:03d}/{epochs} | train={train_loss:.6f} | validation={validation_loss:.6f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "mobile_gru_autoencoder.pt")

    pd.DataFrame(history).to_csv(output_dir / "mobile_training_history.csv", index=False)
    return history


def get_window_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    model.eval()
    window_errors: list[float] = []
    timestep_errors: list[torch.Tensor] = []
    timestep_masks: list[torch.Tensor] = []

    with torch.no_grad():
        for windows, masks in loader:
            windows = windows.to(device)
            masks = masks.to(device)
            reconstruction = forward_batch(model, windows)
            squared_error = (windows - reconstruction).pow(2)
            per_timestep = squared_error.mean(dim=-1)
            # Store unmasked timestep errors; masks are applied during evaluation.
            timestep_errors.append(per_timestep.cpu())
            timestep_masks.append(masks.cpu())
            feature_mask = masks.unsqueeze(-1)
            masked_error = squared_error * feature_mask
            denominator = feature_mask.sum(dim=(1, 2)) * windows.shape[-1]
            scores = masked_error.sum(dim=(1, 2)) / denominator.clamp_min(1.0)
            window_errors.extend(scores.cpu().tolist())

    return window_errors, torch.cat(timestep_errors), torch.cat(timestep_masks)


def make_validation_labels() -> tuple[torch.Tensor, torch.Tensor]:
    label_windows: list[torch.Tensor] = []
    label_masks: list[torch.Tensor] = []
    for video_length, ranges in zip(VALIDATION_LABEL_LENGTHS, VALIDATION_LABEL_RANGES):
        labels = [0] * video_length
        for start, end, label in ranges:
            if start < 0 or end >= video_length or end < start:
                raise ValueError(f"Invalid label range {(start, end, label)} for length {video_length}")
            for timestep in range(start, end + 1):
                labels[timestep] = label
        for start in range(0, len(labels), STRIDE):
            current = torch.tensor(labels[start : start + WINDOW_SIZE], dtype=torch.long)
            real_length = current.numel()
            if real_length < WINDOW_SIZE:
                current = torch.cat([current, torch.zeros(WINDOW_SIZE - real_length, dtype=torch.long)])
            mask = torch.zeros(WINDOW_SIZE, dtype=torch.float32)
            mask[:real_length] = 1.0
            label_windows.append(current)
            label_masks.append(mask)
    return torch.stack(label_windows), torch.stack(label_masks)


def evaluate_validation(
    timestep_errors: torch.Tensor,
    timestep_masks: torch.Tensor,
    label_windows: torch.Tensor,
    label_masks: torch.Tensor,
    output_dir: Path,
) -> dict[str, Any]:
    if timestep_errors.shape != label_windows.shape:
        raise ValueError(
            "Validation label/error shape mismatch: "
            f"errors={tuple(timestep_errors.shape)}, labels={tuple(label_windows.shape)}. "
            "Check validation video order and label lengths before interpreting metrics."
        )

    valid = (timestep_masks > 0) & (label_masks > 0)
    labels = label_windows[valid].cpu().numpy()
    scores = timestep_errors[valid].cpu().numpy()
    binary_labels = np.isin(labels, [1, 2]).astype(np.int64)

    if np.unique(binary_labels).size < 2:
        raise ValueError("Validation metrics require both normal and cheating-labelled timesteps.")

    rows = []
    directions = {
        "higher_error_is_cheating": scores,
        "lower_error_is_cheating": -scores,
    }
    for direction, direction_scores in directions.items():
        auroc = roc_auc_score(binary_labels, direction_scores)
        auprc = average_precision_score(binary_labels, direction_scores)
        precision, recall, thresholds = precision_recall_curve(binary_labels, direction_scores)
        if len(thresholds):
            f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
            best_index = int(np.argmax(f1_values))
            threshold = float(thresholds[best_index])
            best_f1 = float(f1_values[best_index])
            predictions = (direction_scores >= threshold).astype(np.int64)
            balanced_accuracy = balanced_accuracy_score(binary_labels, predictions)
        else:
            threshold, best_f1, balanced_accuracy = float("nan"), float("nan"), float("nan")
        rows.append({
            "direction": direction,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "best_f1_on_validation": best_f1,
            "balanced_accuracy_at_validation_best_f1_threshold": float(balanced_accuracy),
            "threshold_selected_on_validation": threshold,
            "valid_timesteps": int(valid.sum().item()),
            "normal_timesteps": int((binary_labels == 0).sum()),
            "cheating_timesteps": int((binary_labels == 1).sum()),
        })

    pd.DataFrame(rows).to_csv(output_dir / "mobile_validation_metrics.csv", index=False)
    return {"metrics": rows, "valid_timesteps": int(valid.sum().item())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("mobile"), help="Directory containing mobile .pkl files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mobile"), help="Directory for model and CSV outputs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-root", type=Path, default=Path("."), help="Directory containing models/gru.py")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_root = str(args.model_root.resolve())
    if model_root not in sys.path:
        sys.path.insert(0, model_root)

    try:
        from models.gru import MultiLayerGRU
    except ImportError as exc:
        raise ImportError(
            "Could not import MultiLayerGRU. Run from the repository root or set --model-root "
            "to the directory containing models/gru.py."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    filenames, records = load_records(args.data_dir)
    if max(VALIDATION_INDICES + TEST_INDICES) >= len(records):
        raise IndexError(f"The split needs file index {max(VALIDATION_INDICES + TEST_INDICES)}, but only {len(records)} files were found.")

    videos = [extract_video_features(record) for record in records]
    representations = [video["tensor"] for video in videos]

    validation_videos = [representations[index] for index in VALIDATION_INDICES]
    test_videos = [representations[index] for index in TEST_INDICES]
    train_videos = representations[:2] + representations[9:22] + representations[23:]
    if not train_videos:
        raise ValueError("Training split is empty; verify the file count and split indices.")

    train_raw, train_masks, _ = make_windows(train_videos)
    validation_raw, validation_masks, _ = make_windows(validation_videos)
    test_raw, test_masks, _ = make_windows(test_videos)
    train_windows, validation_windows, test_windows, feature_min, feature_max = normalize_windows(
        train_raw, validation_raw, test_raw
    )

    print(f"Train windows: {len(train_windows)} | validation windows: {len(validation_windows)} | test windows: {len(test_windows)}")
    print(f"Model input shape: {tuple(train_windows.shape)} (7 input features; class_id excluded)")
    print("Feature min/max fitted on training data only:")
    for index, name in enumerate(FEATURE_NAMES):
        print(f"  {index} {name}: min={feature_min[index].item():.6f}, max={feature_max[index].item():.6f}")

    train_loader = DataLoader(SequenceDataset(train_windows, train_masks), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(SequenceDataset(validation_windows, validation_masks), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(test_windows, test_masks), batch_size=args.batch_size, shuffle=False)

    model = MultiLayerGRU(
        input_dim=train_windows.shape[-1],
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    train_model(model, train_loader, validation_loader, device, args.epochs, args.learning_rate, args.output_dir)
    checkpoint = args.output_dir / "mobile_gru_autoencoder.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    validation_window_errors, validation_timestep_errors, validation_timestep_masks = get_window_errors(model, validation_loader, device)
    test_window_errors, _, _ = get_window_errors(model, test_loader, device)

    pd.DataFrame({"window_index": np.arange(len(validation_window_errors)), "reconstruction_error": validation_window_errors}).to_csv(args.output_dir / "mobile_validation_window_errors.csv", index=False)
    pd.DataFrame({"window_index": np.arange(len(test_window_errors)), "reconstruction_error": test_window_errors}).to_csv(args.output_dir / "mobile_test_window_errors.csv", index=False)

    test_summary = {
        "modality": "Mobile",
        "model": "MultiLayerGRU reconstruction model",
        "input_features": FEATURE_NAMES[:7],
        "excluded_feature": "class_id",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "test_windows": len(test_window_errors),
        "mean_reconstruction_error": float(np.mean(test_window_errors)),
        "median_reconstruction_error": float(np.median(test_window_errors)),
        "std_reconstruction_error": float(np.std(test_window_errors, ddof=1)) if len(test_window_errors) > 1 else 0.0,
        "min_reconstruction_error": float(np.min(test_window_errors)),
        "max_reconstruction_error": float(np.max(test_window_errors)),
        "intended_use": "Input score for multimodal fusion; not a standalone final cheating decision",
    }
    (args.output_dir / "mobile_test_summary.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"Component": key, "Description": json.dumps(value) if isinstance(value, list) else value} for key, value in test_summary.items()]).to_csv(args.output_dir / "mobile_test_summary.csv", index=False)

    label_windows, label_masks = make_validation_labels()
    if label_windows.shape[0] != validation_windows.shape[0]:
        raise ValueError(
            f"Validation label/window count mismatch: {label_windows.shape[0]} label windows vs "
            f"{validation_windows.shape[0]} model windows. Check the hard-coded label lengths, "
            "video order, and timestep counts before trusting validation metrics."
        )
    validation_metrics = evaluate_validation(
        validation_timestep_errors,
        validation_timestep_masks,
        label_windows,
        label_masks,
        args.output_dir,
    )
    print("Test summary:")
    print(json.dumps(test_summary, indent=2))
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
