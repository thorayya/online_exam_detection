
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision


class ObjectHandFeatureExtractor:

    def __init__(
        self,
        model_path,
        yolo_model="yolo11n.pt",
        white_paper_model_path="/content/drive/MyDrive/Online_Exam_Cheating/white_paper_best.pt",
        device=None
    ):

        self.model_path = model_path
        self.device = device

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

        if self.device is not None:
            self.paper_model.to(self.device)

        # Separate confidence thresholds
        self.BOOK_THRESHOLD = 0.40
        self.PAPER_THRESHOLD = 0.05

        self.paper_model.conf = self.PAPER_THRESHOLD

        # ========================================================
        # CLASS DEFINITIONS
        # ========================================================

        self.classes = {
            0: "person",
            63: "laptop",
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
    # 1. EXTRACT YOLO11 OBJECTS + WHITE PAPER
    # ============================================================

    def extract_objects(self, frame):

        objects = []

        height, width = frame.shape[:2]

        # --------------------------------------------------------
        # YOLO11: PERSON, LAPTOP, CELL PHONE, BOOK
        # --------------------------------------------------------

        results = self.model(
            frame,
            classes=[0, 63, 67, 73],
            verbose=False,
            device=self.device
        )

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(
                    box.conf[0].item()
                )

                # Separate book threshold
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

        # --------------------------------------------------------
        # YOLOv5: WHITE PAPER
        # --------------------------------------------------------

        paper_results = self.paper_model(
            frame,
            size=640
        )

        for detections in paper_results.xyxy:

            for detection in detections:

                confidence = float(detection[4])

                if confidence < self.PAPER_THRESHOLD:
                    continue

                x1, y1, x2, y2 = (
                    detection[:4].cpu().numpy()
                )

                xn1 = x1 / width
                yn1 = y1 / height
                xn2 = x2 / width
                yn2 = y2 / height

                objects.append({

                    "detection_id": len(objects),

                    "class_id": 80,

                    "name": "white-paper",

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
    # 2. EXTRACT ALL HANDS AND 21 LANDMARKS
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

            # Hand bounding box
            xs = [point["x"] for point in landmarks]

            ys = [point["y"] for point in landmarks]

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
    # 3. EXTRACT FEATURES FROM ONE FRAME
    # ============================================================

    def extract_features(
        self,
        frame,
        time_stamp,
        frame_id
    ):

        height, width = frame.shape[:2]

        objects = self.extract_objects(frame)

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
    # 4. CLOSE MEDIAPIPE
    # ============================================================

    def close(self):

        if self.detector is not None:
            self.detector.close()


