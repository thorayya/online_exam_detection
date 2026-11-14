
import os
import cv2
import torch
import shutil
import tempfile
import numpy as np
import mediapipe as mp

from contextlib import nullcontext
from ultralytics import YOLO
from mediapipe.tasks.python import vision

from sam2.build_sam import build_sam2_video_predictor


class ObjectHandFeatureExtractor:

    def __init__(
        self,
        model_path,
        yolo_model="yolo11n.pt",
        white_paper_model_path="/content/drive/MyDrive/Online_Exam_Cheating/white_paper_best.pt",
        sam2_checkpoint="/content/sam2/checkpoints/sam2.1_hiera_small.pt",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
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
        # SAM2 VIDEO PREDICTOR
        # ========================================================

        self.paper_predictor = build_sam2_video_predictor(
            sam2_model_cfg,
            sam2_checkpoint,
            device=self.device
        )

        self.inference_state = None

        self.paper_tracking_started = False

        self.paper_start_frame = None

        self.paper_obj_id = 1

        self.paper_initial_confidence = None

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
    # 2. DETECT FIRST PAPER WITH YOLOv5
    # ============================================================

    def detect_first_paper(self, video_frames_dir):

        frame_names = sorted(
            [
                name for name in os.listdir(video_frames_dir)
                if name.lower().endswith((".jpg", ".jpeg"))
            ],
            key=lambda name: int(os.path.splitext(name)[0])
        )

        for frame_name in frame_names:

            frame_id = int(
                os.path.splitext(frame_name)[0]
            )

            frame_path = os.path.join(
                video_frames_dir,
                frame_name
            )

            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            paper_results = self.paper_model(
                frame,
                size=640
            )

            for detections in paper_results.xyxy:

                if len(detections) == 0:
                    continue

                best_detection = max(
                    detections,
                    key=lambda detection: float(detection[4])
                )

                confidence = float(best_detection[4])

                if confidence < self.PAPER_THRESHOLD:
                    continue

                x1, y1, x2, y2 = (
                    best_detection[:4].cpu().numpy()
                )

                bbox = np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32
                )

                return frame_id, bbox, confidence

        return None, None, None

    # ============================================================
    # 3. INITIALIZE SAM2 WITH FIRST PAPER DETECTION
    # ============================================================

    def initialize_paper_tracking(
        self,
        frame_id,
        bbox,
        confidence
    ):

        self.paper_start_frame = frame_id

        self.paper_initial_confidence = confidence

        with torch.inference_mode(), self._autocast_context():

            _, out_obj_ids, out_mask_logits = (
                self.paper_predictor.add_new_points_or_box(

                    inference_state=self.inference_state,

                    frame_idx=frame_id,

                    obj_id=self.paper_obj_id,

                    box=bbox

                )
            )

        self.paper_tracking_started = True

        print(
            f"Paper detected at frame {frame_id}. "
            f"SAM2 tracking initialized."
        )

    # ============================================================
    # 4. PROPAGATE PAPER THROUGH VIDEO
    # ============================================================

    def track_paper(self):

        video_segments = {}

        if not self.paper_tracking_started:

            print("No paper detected in the video.")

            return video_segments

        with torch.inference_mode(), self._autocast_context():

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits
            ) in self.paper_predictor.propagate_in_video(
                self.inference_state
            ):

                for i, out_obj_id in enumerate(out_obj_ids):

                    if out_obj_id != self.paper_obj_id:
                        continue

                    mask = (
                        out_mask_logits[i] > 0.0
                    ).squeeze().cpu().numpy()

                    video_segments[out_frame_idx] = mask

        return video_segments

    # ============================================================
    # 5. CONVERT SAM2 MASK TO PAPER OBJECT
    # ============================================================

    def mask_to_paper_object(
        self,
        mask,
        detection_id,
        frame_id
    ):

        if mask is None or not np.any(mask):
            return None

        height, width = mask.shape

        ys, xs = np.where(mask)

        x1 = int(xs.min())
        y1 = int(ys.min())

        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        xn1 = x1 / width
        yn1 = y1 / height

        xn2 = x2 / width
        yn2 = y2 / height

        confidence = (
            self.paper_initial_confidence
            if frame_id == self.paper_start_frame
            else None
        )

        return {

            "detection_id": detection_id,

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

        }

    # ============================================================
    # 6. EXTRACT ALL HANDS AND 21 LANDMARKS
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
    # 7. EXTRACT FEATURES FROM ONE FRAME
    # ============================================================

    def extract_features(
        self,
        frame,
        time_stamp,
        frame_id,
        paper_mask=None
    ):

        height, width = frame.shape[:2]

        objects = self.extract_objects(frame)

        # Add paper position from SAM2 mask
        if paper_mask is not None:

            paper_object = self.mask_to_paper_object(
                paper_mask,
                detection_id=len(objects),
                frame_id=frame_id
            )

            if paper_object is not None:
                objects.append(paper_object)

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
    # 8. PROCESS COMPLETE AVI VIDEO
    # ============================================================

    def process_video(self, video_path):

        if self.inference_state is not None:

            self.paper_predictor.reset_state(
                self.inference_state
            )

            self.inference_state = None

        self.paper_tracking_started = False

        self.paper_start_frame = None

        self.paper_initial_confidence = None

        video_frames_dir = tempfile.mkdtemp(
            prefix="sam2_video_frames_"
        )

        frame_features_list = []

        video_segments = {}

        try:

            # ----------------------------------------------------
            # Convert AVI to numbered JPG frames
            # ----------------------------------------------------

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot open video: {video_path}"
                )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25.0

            frame_id = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_path = os.path.join(
                    video_frames_dir,
                    f"{frame_id:06d}.jpg"
                )

                if not cv2.imwrite(frame_path, frame):
                    raise RuntimeError(
                        f"Could not save frame {frame_id}"
                    )

                frame_id += 1

            cap.release()

            if frame_id == 0:
                raise RuntimeError(
                    "No frames were extracted from the AVI."
                )

            print(f"Extracted {frame_id} frames.")

            # ----------------------------------------------------
            # Initialize SAM2 video state
            # ----------------------------------------------------

            self.inference_state = (
                self.paper_predictor.init_state(
                    video_path=video_frames_dir
                )
            )

            # ----------------------------------------------------
            # Find first valid paper detection
            # ----------------------------------------------------

            first_frame_id, first_bbox, confidence = (
                self.detect_first_paper(video_frames_dir)
            )

            # ----------------------------------------------------
            # Initialize tracking at first detection
            # ----------------------------------------------------

            if first_frame_id is not None:

                self.initialize_paper_tracking(
                    first_frame_id,
                    first_bbox,
                    confidence
                )

                # ------------------------------------------------
                # Track paper from its first detection onward
                # ------------------------------------------------

                video_segments = self.track_paper()

            else:

                print(
                    "No valid paper detection found. "
                    "SAM2 tracking was not started."
                )

            # ----------------------------------------------------
            # Extract features from every original frame
            # ----------------------------------------------------

            frame_names = sorted(
                [
                    name for name in os.listdir(video_frames_dir)
                    if name.lower().endswith((".jpg", ".jpeg"))
                ],
                key=lambda name: int(
                    os.path.splitext(name)[0]
                )
            )

            for frame_name in frame_names:

                current_frame_id = int(
                    os.path.splitext(frame_name)[0]
                )

                frame_path = os.path.join(
                    video_frames_dir,
                    frame_name
                )

                frame = cv2.imread(frame_path)

                if frame is None:
                    continue

                # MediaPipe timestamp in milliseconds
                time_stamp = int(
                    current_frame_id * 1000 / fps
                )

                paper_mask = video_segments.get(
                    current_frame_id
                )

                features = self.extract_features(
                    frame=frame,
                    time_stamp=time_stamp,
                    frame_id=current_frame_id,
                    paper_mask=paper_mask
                )

                frame_features_list.append(features)

            return frame_features_list, video_segments

        finally:

            shutil.rmtree(
                video_frames_dir,
                ignore_errors=True
            )

    # ============================================================
    # 9. CLOSE MODELS
    # ============================================================

    def close(self):

        if self.inference_state is not None:

            self.paper_predictor.reset_state(
                self.inference_state
            )

            self.inference_state = None

        if self.detector is not None:
            self.detector.close()


        if self.detector is not None:
            self.detector.close()


