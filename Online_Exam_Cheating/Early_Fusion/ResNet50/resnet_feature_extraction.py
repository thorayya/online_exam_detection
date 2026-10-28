
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, ResNetModel


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "/content/oep_dataset/OEP database"
OUTPUT_DIR = Path("/content")

VIDEO_INDEX = 1
SUBJECT_INDEX = 16

SAMPLE_EVERY = 25
BATCH_SIZE = 16

MODEL_NAME = "microsoft/resnet-50"


# ==========================================
# 2. DEVICE AND RESNET-50
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = ResNetModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False


# ==========================================
# 3. COLLECT VIDEO PATHS
# ==========================================

video_paths = []

for subject in sorted(os.listdir(DATASET_PATH)):
    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    videos = sorted(
        filename
        for filename in os.listdir(subject_path)
        if filename.lower().endswith(".avi")
    )

    if len(videos) > VIDEO_INDEX:
        selected_video = videos[VIDEO_INDEX]

        video_paths.append({
            "subject": subject,
            "video": selected_video,
            "path": os.path.join(subject_path, selected_video)
        })

if len(video_paths) <= SUBJECT_INDEX:
    raise IndexError(
        f"Video index {SUBJECT_INDEX} is unavailable. "
        f"Found {len(video_paths)} eligible videos."
    )

video_path = video_paths[SUBJECT_INDEX]["path"]
video_name = Path(video_path).stem

print("Selected video:", video_path)


# ==========================================
# 4. LABEL INTERVALS
# Format: (start_seconds, end_seconds, class_id)
# ==========================================

labels = [
    (27, 36, 1),
    (39, 50, 1),
    (169, 189, 1),
    (294, 315, 5),
    (321, 322, 5),
    (388, 404, 1),
    (523, 524, 2),
    (528, 529, 2),
    (534, 543, 2),
    (563, 577, 1),
    (855, 856, 2),
    (858, 859, 2),
]

suspicious_intervals = [
    (start, end, label)
    for start, end, label in labels
    if label in (1, 3, 5, 6)
]

cheating_intervals = [
    (start, end, label)
    for start, end, label in labels
    if label in (2, 4)
]


# ==========================================
# 5. VIDEO AND EXTRACTION SETUP
# ==========================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

frame_count = 0
sampled_count = 0

frame_features = []
frame_labels = []

batch_images = []
batch_labels = []


# ==========================================
# 6. PROCESS ONE BATCH
# ==========================================

def process_batch(images, metadata):
    if not images:
        return

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

        # Expected shape: (B, 2048, 7, 7)
        feature_maps = outputs.last_hidden_state

        # Shape: (B, 2048, 7, 7) -> (B, 100352)
        features = feature_maps.flatten(start_dim=1)

    frame_features.extend(
        features.cpu().numpy().astype(np.float32)
    )

    frame_labels.extend(metadata)


# ==========================================
# 7. READ VIDEO AND EXTRACT FEATURES
# ==========================================

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Sample one frame every 25 original frames
        if frame_count % SAMPLE_EVERY != 0:
            continue

        sampled_count += 1

        # Timestamp in seconds
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        suspicious = any(
            start <= timestamp <= end
            for start, end, _ in suspicious_intervals
        )

        cheating = any(
            start <= timestamp <= end
            for start, end, _ in cheating_intervals
        )

        # Normal if neither suspicious nor cheating
        normal = not suspicious and not cheating

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        batch_images.append(image)

        batch_labels.append({
            "frame_number": frame_count,
            "timestamp": timestamp,
            "normal": int(normal),
            "suspicious": int(suspicious),
            "cheating": int(cheating)
        })

        if len(batch_images) >= BATCH_SIZE:
            process_batch(batch_images, batch_labels)

            batch_images = []
            batch_labels = []

            print(
                f"Processed {len(frame_features)} sampled images "
                f"(original frame: {frame_count})"
            )

    # Process the final incomplete batch
    process_batch(batch_images, batch_labels)

finally:
    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# 8. VALIDATE RESULTS
# ==========================================

frame_features = np.asarray(
    frame_features,
    dtype=np.float32
)

print("\nTotal original frames:", frame_count)
print("Sampled images:", sampled_count)
print("Feature matrix shape:", frame_features.shape)
print("Labeled images:", len(frame_labels))

print(
    "Label counts:",
    {
        "normal": sum(x["normal"] for x in frame_labels),
        "suspicious": sum(x["suspicious"] for x in frame_labels),
        "cheating": sum(x["cheating"] for x in frame_labels)
    }
)

print(
    "First image label:",
    frame_labels[0] if frame_labels else None
)

if len(frame_features) > 0:
    print("Features per image:", frame_features.shape[1])

    assert len(frame_features) == len(frame_labels)

    assert frame_features.shape[1] == 100352, (
        "Unexpected feature dimensions; "
        "inspect feature_maps.shape."
    )

    print("Feature and label counts match.")


# ==========================================
# 9. SAVE RESNET OUTPUTS
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features_path = OUTPUT_DIR / f"{video_name}_features.npy"
labels_path = OUTPUT_DIR / f"{video_name}_labels.json"

np.save(features_path, frame_features)

with open(labels_path, "w", encoding="utf-8") as file:
    json.dump(
        frame_labels,
        file,
        ensure_ascii=False,
        indent=2
    )

print("\nFeatures saved:", features_path)
print("Labels saved:", labels_path)
print("Feature shape:", frame_features.shape)
print("Number of saved labels:", len(frame_labels))
