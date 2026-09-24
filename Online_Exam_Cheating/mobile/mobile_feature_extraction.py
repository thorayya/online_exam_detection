import os
import cv2
import pickle

from mobile.object_hand_features import ObjectHandFeatureExtractor


# Path to the dataset
dataset_path = "PATH_TO_OEP_DATABASE"

video_paths = []

for subject in sorted(os.listdir(dataset_path)):

    subject_path = os.path.join(dataset_path, subject)

    if os.path.isdir(subject_path):

        videos = [
            f for f in os.listdir(subject_path)
            if f.endswith(".avi")
        ]

        if videos:
            selected_video = videos[0]

            video_paths.append({
                "subject": subject,
                "video": selected_video,
                "path": os.path.join(subject_path, selected_video)
            })


# Path to MediaPipe hand landmark model
model = "hand_landmarker.task"

extractor = ObjectHandFeatureExtractor(
    model_path=model
)

all_sequences = []


for item in video_paths:

    cap = cv2.VideoCapture(item["path"])

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1
        time_stamp = int((frame_id / fps) * 1000)

        if frame_id % 25 == 0:

            try:
                prohibited_objects = extractor.extract_features(
                    frame,
                    time_stamp,
                    frame_id
                )

            except Exception as e:
                print(
                    f"Feature extraction error at frame {frame_id}:",
                    e
                )
                prohibited_objects = None

            if frame_id % 500 == 0:
                print(f"Processed {frame_id} frames")

            all_sequences.append({
                "subject": item["subject"],
                "video": item["video"],
                "frame_id": frame_id,
                "time_stamp": time_stamp,
                "objects": prohibited_objects
            })

    cap.release()



output_path = "mobile_features.pkl"

with open(output_path, "wb") as f:
    pickle.dump(all_sequences, f)

print(f"Saved extracted features to: {output_path}")


