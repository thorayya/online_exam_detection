

import time
from huggingface_hub import hf_hub_download
import cv2
from vision.eye_tracking import EyeTrackingGaze
from vision.objects_detection import ObjectDetection
from vision.head_pose import HeadPoseExtractor
from vision.face_detection import FaceDetector


# import os


# os.environ["KAGGLE_API_TOKEN"] = 


# dataset_path = "OEP database"

# video_paths = []

# for subject in sorted(os.listdir(dataset_path)):

#     subject_path = os.path.join(dataset_path, subject)

#     if os.path.isdir(subject_path):

#         videos = [
#             f for f in os.listdir(subject_path)
#             if f.endswith(".avi")
#         ]

#         if len(videos) >= 1:
#             selected_video = videos[1]

#             video_paths.append({
#                 "subject": subject,
#                 "video": selected_video,
#                 "path": os.path.join(subject_path, selected_video)
#             })






model_asset_path = "face_landmarker_v2_with_blendshapes.task"


detector = ObjectDetection(
    model_path="pose_landmarker.task",
    yolo_model_path="yolov8s.pt",
    earphone_model_path="best.pt"
)


weights = hf_hub_download(
    repo_id="tianfxc/l2cs",
    filename="L2CSNet_gaze360.pkl"
)

tracker = EyeTrackingGaze(
    model_asset_path=model_asset_path,
    weights=weights )

face_detector = FaceDetector(
    model_path="detector.tflite",
)


pose_extractor = HeadPoseExtractor(
    "face_landmarker_v2_with_blendshapes.task"
)

all_sequences = []
objects_boxes = []
dets = []
gaze_result = None
pose = None



for item in video_paths[23:]:

    video_path = item["path"]
    video_id = item["subject"]

    sequence_data = {
        "video_id": video_id,
        "frames": []
    }

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    frame_id = 0



    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))



        try:

          face_result = face_detector.face_detection(
              int(timestamp_ms),
              frame
          )
        except Exception as e:
          print(
              f"🔴Face detection error at frame {frame_id}:",
              e
          )

        try:
          head_pose = pose_extractor.extract(
              frame,
              timestamp_ms
          )
        except Exception as e:
          print(
              f"🔴Head pose error at frame {frame_id}:",
              e
          )

        if frame_id % 10 == 0:
          try:

              pose = detector.pose_detection(
                  frame,
                  timestamp_ms,
                  w,
                  h
              )


          except Exception as e:

              print(
                  f"🔴Pose error at frame {frame_id}:",
                  e
              )



        if frame_id % 25 == 0:
          try:

              objects_boxes, dets = detector.object_detection(
                  frame
              )

          except Exception as e:

              print(
                  f"🔴Object detection error at frame {frame_id}:",
                  e
              )


        if frame_id % 25 == 0:
          try:

              gaze_result = tracker.extract(
                  frame,
                  timestamp_ms
              )

          except Exception as e:

              print(
                  f"🔴Gaze error at frame {frame_id}:",
                  e
              )



        frame_data = {

            "frame_id": frame_id,

            "timestamp_ms": timestamp_ms,

            "face": face_result,

            "gaze": gaze_result,

            "head_pose": head_pose,

            "objects": [],

            "pose_features": {

                "left_elbow_angle": None,

                "right_elbow_angle": None,

                "left_wrist_distance": None,

                "right_wrist_distance": None,

                "left_elbow_distance": None,

                "right_elbow_distance": None
            }
        }



        for det in dets:

            frame_data["objects"].append({

                "bbox": [
                    float(det[0]),
                    float(det[1]),
                    float(det[2]),
                    float(det[3])
                ],

                "conf": float(det[4]),

                "class": int(det[5])
            })

        if pose is not None:

            try:

                left_elbow_angle = detector.angle(
                    pose["left_shoulder"],
                    pose["left_elbow"],
                    pose["left_wrist"]
                )

                right_elbow_angle = detector.angle(
                    pose["right_shoulder"],
                    pose["right_elbow"],
                    pose["right_wrist"]
                )

                features = detector.build_features(
                    pose,
                    objects_boxes
                )

                frame_data["pose_features"] = {

                    "left_elbow_angle":
                        float(left_elbow_angle),

                    "right_elbow_angle":
                        float(right_elbow_angle),

                    "left_wrist_distance":
                        None if features is None
                        else float(features[0]),

                    "right_wrist_distance":
                        None if features is None
                        else float(features[1]),

                    "left_elbow_distance":
                        None if features is None
                        else float(features[2]),

                    "right_elbow_distance":
                        None if features is None
                        else float(features[3])
                }

            except Exception as e:

                print(
                    f"🔴Pose feature error at frame {frame_id}:",
                    e
                )




        sequence_data["frames"].append(
            frame_data
        )

        frame_id += 1

    cap.release()



    all_sequences.append(
        sequence_data
    )


print(
    f"Frames in first sequence: "
    f"{len(all_sequences[0]['frames'])}"
    if all_sequences
    else "No sequences extracted."
)



import pickle

with open("/chenlipi.pkl", "wb") as f:
    pickle.dump(all_sequences, f)
