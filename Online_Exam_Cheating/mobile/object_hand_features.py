
import os
import cv2
import pickle
import numpy as np

from collections import Counter

from mobile.object_hand_features import (
    ObjectHandFeatureExtractor
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

dataset_path = "PATH_TO_OEP_DATABASE"

model_path = "hand_landmarker.task"

yolo_model = "yolo11n.pt"

output_path = (
    "/content/drive/MyDrive/"
    "Online_Exam_Cheating/mobile/"
    "mobile_features_full.pkl"
)

FRAME_INTERVAL = 25

DEVICE = None
# None = automatic device selection by Ultralytics


# ============================================================
# 2. VERIFY PATHS
# ============================================================

if not os.path.isdir(dataset_path):

    raise FileNotFoundError(
        f"Dataset directory not found: {dataset_path}"
    )

if not os.path.isfile(model_path):

    raise FileNotFoundError(
        f"Hand landmark model not found: {model_path}"
    )

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)


# ============================================================
# 3. FIND VIDEOS
# ============================================================

video_paths = []

for subject in sorted(os.listdir(dataset_path)):

    subject_path = os.path.join(
        dataset_path,
        subject
    )

    if not os.path.isdir(subject_path):
        continue

    videos = sorted([

        f for f in os.listdir(subject_path)

        if f.lower().endswith(".avi")

    ])

    if not videos:
        continue

    # Preserve the original selection logic:
    # first AVI video in each subject folder.

    selected_video = videos[0]

    video_paths.append({

        "subject": subject,

        "video": selected_video,

        "path": os.path.join(
            subject_path,
            selected_video
        )

    })


print("=" * 70)
print("VIDEO DISCOVERY")
print("=" * 70)

print("Videos found:", len(video_paths))

for item in video_paths:

    print(
        item["subject"],
        "|",
        item["video"]
    )


# ============================================================
# 4. INITIALIZE EXTRACTOR
# ============================================================

extractor = ObjectHandFeatureExtractor(

    model_path=model_path,

    yolo_model=yolo_model,

    device=DEVICE

)


# ============================================================
# 5. PROCESS VIDEOS
# ============================================================

all_sequences = []

total_frames = 0

total_objects = 0

total_hands = 0

total_relations = 0

error_count = 0

video_summaries = []


try:

    for video_number, item in enumerate(
        video_paths,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"VIDEO {video_number}/{len(video_paths)}:",
            item["subject"],
            item["video"]
        )

        print("=" * 70)

        cap = cv2.VideoCapture(item["path"])

        if not cap.isOpened():

            print(
                "WARNING: Cannot open video:",
                item["path"]
            )

            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:

            print(
                "WARNING: Invalid FPS. Using 25."
            )

            fps = 25.0

        frame_id = 0

        video_frames = 0

        video_objects = 0

        video_hands = 0

        video_relations = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            # Process one frame every 25 frames
            if frame_id % FRAME_INTERVAL != 0:
                continue

            # Frame timestamp in milliseconds.
            # Using frame_id - 1 follows zero-based frame timing.

            time_stamp = int(
                ((frame_id - 1) / fps) * 1000
            )

            try:

                frame_result = (
                    extractor.extract_features(

                        frame=frame,

                        time_stamp=time_stamp,

                        frame_id=frame_id

                    )
                )

                frame_record = {

                    "subject": item["subject"],

                    "video": item["video"],

                    "frame_id": frame_id,

                    "time_stamp": time_stamp,

                    "fps": float(fps),

                    "objects": frame_result["objects"],

                    "hands": frame_result["hands"],

                    "hand_object_relations": (
                        frame_result[
                            "hand_object_relations"
                        ]
                    ),

                    "object_count": (
                        frame_result["object_count"]
                    ),

                    "hand_count": (
                        frame_result["hand_count"]
                    ),

                    "relation_count": (
                        frame_result["relation_count"]
                    ),

                    "image_width": (
                        frame_result["image_width"]
                    ),

                    "image_height": (
                        frame_result["image_height"]
                    ),

                    "extraction_error": None

                }

            except Exception as e:

                error_count += 1

                print(
                    f"ERROR | {item['subject']} | "
                    f"Frame {frame_id} | {e}"
                )

                frame_record = {

                    "subject": item["subject"],

                    "video": item["video"],

                    "frame_id": frame_id,

                    "time_stamp": time_stamp,

                    "fps": float(fps),

                    "objects": [],

                    "hands": [],

                    "hand_object_relations": [],

                    "object_count": 0,

                    "hand_count": 0,

                    "relation_count": 0,

                    "image_width": int(frame.shape[1]),

                    "image_height": int(frame.shape[0]),

                    "extraction_error": str(e)

                }

            all_sequences.append(frame_record)

            video_frames += 1

            video_objects += frame_record["object_count"]

            video_hands += frame_record["hand_count"]

            video_relations += frame_record["relation_count"]

            if frame_record["extraction_error"] is None:

                total_objects += frame_record["object_count"]

                total_hands += frame_record["hand_count"]

                total_relations += frame_record["relation_count"]

            if video_frames % 100 == 0:

                print(

                    f"Processed sampled frames: {video_frames} | "

                    f"Objects: {video_objects} | "

                    f"Hands: {video_hands} | "

                    f"Relations: {video_relations}"

                )

        cap.release()

        total_frames += video_frames

        video_summaries.append({

            "subject": item["subject"],

            "video": item["video"],

            "fps": float(fps),

            "sampled_frames": video_frames,

            "objects": video_objects,

            "hands": video_hands,

            "relations": video_relations

        })

        print("\nVIDEO SUMMARY")

        print("Sampled frames:", video_frames)

        print("Objects:", video_objects)

        print("Hands:", video_hands)

        print("Hand-object relations:", video_relations)


finally:

    extractor.close()


# ============================================================
# 6. SAVE DATA
# ============================================================

with open(output_path, "wb") as f:

    pickle.dump(
        all_sequences,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# 7. FINAL AUDIT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EXTRACTION AUDIT")
print("=" * 70)

print("Videos:", len(video_summaries))

print("Sampled frames:", total_frames)

print("Objects:", total_objects)

print("Hands:", total_hands)

print("Hand-object relations:", total_relations)

print("Extraction errors:", error_count)


# Object class distribution

class_counter = Counter()

for frame in all_sequences:

    if frame["extraction_error"] is not None:
        continue

    for obj in frame["objects"]:

        class_counter[obj["name"]] += 1


print("\nOBJECT CLASS DISTRIBUTION")

for name, count in class_counter.most_common():

    print(f"{name:20s}: {count}")


# Handedness distribution

hand_counter = Counter()

for frame in all_sequences:

    if frame["extraction_error"] is not None:
        continue

    for hand in frame["hands"]:

        hand_counter[hand["handedness"]] += 1


print("\nHAND DISTRIBUTION")

for name, count in hand_counter.items():

    print(f"{name:20s}: {count}")


# Landmark verification

valid_hand_records = [

    hand

    for frame in all_sequences

    if frame["extraction_error"] is None

    for hand in frame["hands"]

]

if valid_hand_records:

    landmark_counts = Counter(

        len(hand["landmarks"])

        for hand in valid_hand_records

    )

    print("\nLANDMARK COUNT AUDIT")

    for count, frequency in landmark_counts.items():

        print(
            f"{count} landmarks: {frequency} hands"
        )

    assert all(
        len(hand["landmarks"]) == 21
        for hand in valid_hand_records
    ), "Some hands do not have exactly 21 landmarks."


print("\nVIDEO SUMMARIES")

for summary in video_summaries:

    print(summary)


print("\nSaved:", output_path)

print("Extraction completed.")
