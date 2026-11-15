
import os
import cv2
import torch
import numpy as np
import mediapipe as mp

from argparse import Namespace
from contextlib import nullcontext
from ultralytics import YOLO
from mediapipe.tasks.python import vision

from yolox.tracker.byte_tracker import BYTETracker


class ObjectHandFeatureExtractor:

    def __init__(
        self,
        model_path,
        yolo_model="yolo11n.pt",
        white_paper_model_path="/content/drive/MyDrive/Online_Exam_Cheating/white_paper_best.pt",
        device=None
    ):

        self.model_path = model_path

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ========================================================
        # YOLO11 MODEL
        # ========================================================

        self.model = YOLO(yolo_model)

        # ========================================================
        # CUSTOM YOLOv5 MODEL - WHITE PAPER
        # ========================================================

        self.paper_model = torch.hub.load(
            "/content/yolov5",
            "custom",
            path=white_paper_model_path,
            source="local"
        )

        self.paper_model.to(self.device)

        self.BOOK_THRESHOLD = 0.40
        self.PAPER_THRESHOLD = 0.05

        self.paper_model.conf = self.PAPER_THRESHOLD

        # ========================================================
        # BYTE TRACKER
        # ========================================================

        tracker_args = Namespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False
        )

        self.paper_tracker = BYTETracker(
            tracker_args,
            frame_rate=25
        )

        # ========================================================
        # CLASS DEFINITIONS
        # ========================================================

        self.classes = {
            67: "cell phone",
            73: "book",
            80: "white-paper"
        }

        # ========================================================
        # MEDIAPIPE INITIALIZATION
        # ========================================================

        BaseOptions = mp.tasks.BaseOptions

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        VisionRunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),

            running_mode=VisionRunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5

        )

        self.detector = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        # ========================================================
        # LANDMARK NAMES
        # ========================================================

        self.landmark_names = [

            "wrist",

            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",

            "index_mcp",
            "index_pip",
            "index_dip",
            "index_tip",

            "middle_mcp",
            "middle_pip",
            "middle_dip",
            "middle_tip",

            "ring_mcp",
            "ring_pip",
            "ring_dip",
            "ring_tip",

            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip"

        ]

    # ============================================================
    # AUTocast CONTEXT
    # ============================================================

    def _autocast_context(self):

        if not str(self.device).startswith("cuda"):
            return nullcontext()

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        return torch.autocast(
            device_type="cuda",
            dtype=dtype
        )

    # ============================================================
    # 1. EXTRACT YOLO11 OBJECTS
    # ============================================================

    def extract_objects(self, frame):

        objects = []

        height, width = frame.shape[:2]

        results = self.model(
            frame,
            classes=[67, 73],
            verbose=False,
            device=self.device
        )

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(
                    box.conf[0].item()
                )

                if (
                    class_id == 73
                    and confidence < self.BOOK_THRESHOLD
                ):
                    continue

                bbox_xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                bbox_normalized = (
                    box.xyxyn[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                x1, y1, x2, y2 = bbox_xyxy

                xn1, yn1, xn2, yn2 = bbox_normalized

                objects.append({

                    "detection_id": len(objects),

                    "class_id": class_id,

                    "name": self.classes[class_id],

                    "conf": confidence,

                    "bbox_xyxy": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],

                    "bbox_normalized": [
                        float(xn1),
                        float(yn1),
                        float(xn2),
                        float(yn2)
                    ],

                    "center_normalized": [
                        float((xn1 + xn2) / 2),
                        float((yn1 + yn2) / 2)
                    ],

                    "width_normalized": float(xn2 - xn1),

                    "height_normalized": float(yn2 - yn1)

                })

        return objects

    # ============================================================
    # 2. DETECT PAPER WITH YOLOv5 IN EVERY FRAME
    # ============================================================

    def detect_paper(self, frame):

        paper_results = self.paper_model(
            frame,
            size=640
        )

        detections = paper_results.xyxy[0]

        if len(detections) == 0:
            return np.empty((0, 5), dtype=np.float32)

        detections = (
            detections[:, :5]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return detections

    # ============================================================
    # 3. TRACK PAPER WITH BYTE TRACK
    # ============================================================

    def track_paper(self, frame):

        height, width = frame.shape[:2]

        detections = self.detect_paper(frame)

        online_targets = self.paper_tracker.update(
            detections,
            (height, width),
            (height, width)
        )

        paper_objects = []

        for track in online_targets:

            bbox = track.tlbr

            paper_object = self.bbox_to_paper_object(
                bbox=bbox,
                confidence=float(track.score),
                track_id=int(track.track_id),
                detection_id=len(paper_objects),
                frame_width=width,
                frame_height=height
            )

            paper_objects.append(paper_object)

        return paper_objects

    # ============================================================
    # 4. CONVERT BYTE TRACK BBOX TO PAPER OBJECT
    # ============================================================

    def bbox_to_paper_object(
        self,
        bbox,
        confidence,
        track_id,
        detection_id,
        frame_width,
        frame_height
    ):

        x1, y1, x2, y2 = bbox

        x1 = float(np.clip(x1, 0, frame_width))
        y1 = float(np.clip(y1, 0, frame_height))
        x2 = float(np.clip(x2, 0, frame_width))
        y2 = float(np.clip(y2, 0, frame_height))

        xn1 = x1 / frame_width
        yn1 = y1 / frame_height

        xn2 = x2 / frame_width
        yn2 = y2 / frame_height

        return {

            "detection_id": detection_id,

            "track_id": track_id,

            "class_id": 80,

            "name": "white-paper",

            "conf": confidence,

            "bbox_xyxy": [
                x1,
                y1,
                x2,
                y2
            ],

            "bbox_normalized": [
                float(xn1),
                float(yn1),
                float(xn2),
                float(yn2)
            ],

            "center_normalized": [
                float((xn1 + xn2) / 2),
                float((yn1 + yn2) / 2)
            ],

            "width_normalized": float(xn2 - xn1),

            "height_normalized": float(yn2 - yn1)

        }

    # ============================================================
    # 5. EXTRACT ALL HANDS AND 21 LANDMARKS
    # ============================================================

    def extract_hands(self, frame, time_stamp):

        hands = []

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect_for_video(
            mp_image,
            time_stamp
        )

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[hand_index][0]
                .category_name
            )

            handedness_score = float(
                result.handedness[hand_index][0]
                .score
            )

            landmarks = []

            for landmark_index, landmark in enumerate(
                hand_landmarks
            ):

                landmarks.append({

                    "index": landmark_index,

                    "name": self.landmark_names[
                        landmark_index
                    ],

                    "x": float(landmark.x),

                    "y": float(landmark.y),

                    "z": float(landmark.z),

                    "x_pixel": float(
                        landmark.x * width
                    ),

                    "y_pixel": float(
                        landmark.y * height
                    )

                })

            xs = [
                point["x"] for point in landmarks
            ]

            ys = [
                point["y"] for point in landmarks
            ]

            hand_bbox = {

                "x_min": float(min(xs)),

                "y_min": float(min(ys)),

                "x_max": float(max(xs)),

                "y_max": float(max(ys))

            }

            hands.append({

                "hand_id": hand_index,

                "handedness": handedness,

                "handedness_score": handedness_score,

                "landmarks": landmarks,

                "landmark_count": len(landmarks),

                "bbox_normalized": hand_bbox

            })

        return hands

    # ============================================================
    # 6. EXTRACT FEATURES FROM ONE FRAME
    # ============================================================

    def extract_features(
        self,
        frame,
        time_stamp,
        frame_id,
        paper_objects=None
    ):

        height, width = frame.shape[:2]

        objects = self.extract_objects(frame)

        # Add ByteTrack paper objects
        if paper_objects is not None:
            objects.extend(paper_objects)

        hands = self.extract_hands(
            frame,
            time_stamp
        )

        frame_features = {

            "frame_id": frame_id,

            "time_stamp": time_stamp,

            "image_width": width,

            "image_height": height,

            "objects": objects,

            "hands": hands,

            "object_count": len(objects),

            "hand_count": len(hands)

        }

        return frame_features

    # ============================================================
    # 7. PROCESS COMPLETE AVI VIDEO
    # ============================================================

    def process_video(self, video_path):

        # Reset ByteTrack for a new video
        tracker_args = Namespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False
        )

        self.paper_tracker = BYTETracker(
            tracker_args,
            frame_rate=25
        )

        frame_features_list = []

        paper_tracks_by_frame = {}

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        frame_id = 0

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # ------------------------------------------------
                # Timestamp in milliseconds
                # ------------------------------------------------

                time_stamp = int(
                    frame_id * 1000 / fps
                )

                # ------------------------------------------------
                # Detect and track paper in every frame
                # ------------------------------------------------

                paper_objects = self.track_paper(
                    frame
                )

                paper_tracks_by_frame[frame_id] = (
                    paper_objects
                )

                # ------------------------------------------------
                # Extract YOLO11 and MediaPipe features
                # ------------------------------------------------

                features = self.extract_features(
                    frame=frame,
                    time_stamp=time_stamp,
                    frame_id=frame_id,
                    paper_objects=paper_objects
                )

                frame_features_list.append(features)

                frame_id += 1

            if frame_id == 0:
                raise RuntimeError(
                    "No frames were extracted from the AVI."
                )

            print(f"Processed {frame_id} frames.")

            return (
                frame_features_list,
                paper_tracks_by_frame
            )

        finally:

            cap.release()

    # ============================================================
    # 8. CLOSE MODELS
    # ============================================================

    def close(self):

        if self.detector is not None:
            self.detector.close()

        if self.detector is not None:
            self.detector.close()


