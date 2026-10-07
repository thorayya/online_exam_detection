from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision


class ObjectHandFeatureExtractor:

    def __init__(self, model_path):

        self.model_path = model_path

        self.model = YOLO("yolo11n.pt")

        self.classes = {
            '0': 'person',
            '63': 'laptop',
            '67': 'cell phone',
            '73': 'book'
        }

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=VisionRunningMode.VIDEO
        )

        self.detector = vision.HandLandmarker.create_from_options(options)


    def extract_features(self, frame, time_stamp, frame_id):

        # Features of the current frame only
        frame_features = []

        # Reset hand points for every frame
        wrist_right_point = None
        wrist_left_point = None

        thumb_tip_right_point = None
        thumb_tip_left_point = None

        finger_tip_right_point = None
        finger_tip_left_point = None


        # ============================================================
        # 1. YOLO object detection
        # ============================================================

        results = self.model(
            frame,
            classes=[0, 63, 67, 73],
            verbose=False
        )


        # ============================================================
        # 2. MediaPipe hand detection
        # ============================================================

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


        # ============================================================
        # 3. Extract hand landmarks
        # ============================================================

        for i, hand in enumerate(result.hand_landmarks):

            handedness = result.handedness[i][0].category_name


            if handedness == "Right":

                wrist_right_point = np.array([
                    float(hand[0].x),
                    float(hand[0].y)
                ])

                finger_tip_right_point = np.array([
                    float(hand[8].x),
                    float(hand[8].y)
                ])

                thumb_tip_right_point = np.array([
                    float(hand[4].x),
                    float(hand[4].y)
                ])


            elif handedness == "Left":

                wrist_left_point = np.array([
                    float(hand[0].x),
                    float(hand[0].y)
                ])

                finger_tip_left_point = np.array([
                    float(hand[8].x),
                    float(hand[8].y)
                ])

                thumb_tip_left_point = np.array([
                    float(hand[4].x),
                    float(hand[4].y)
                ])


        # ============================================================
        # 4. Calculate hand-object distances
        # ============================================================

        for result_yolo in results:

            for i in range(len(result_yolo.boxes)):

                class_id = int(result_yolo.boxes.cls[i])

                if class_id in [int(x) for x in self.classes]:

                    xyxyn = result_yolo.boxes.xyxyn[i].cpu().numpy()

                    conf = result_yolo.boxes.conf[i].cpu().item()

                    name = result_yolo.names[class_id]


                    object_center = np.array([
                        (float(xyxyn[0]) + float(xyxyn[2])) / 2,
                        (float(xyxyn[1]) + float(xyxyn[3])) / 2
                    ], dtype=np.float32)


                    # ------------------------------------------------
                    # Initialize distances
                    # ------------------------------------------------

                    distance_wrist_right = None
                    distance_wrist_left = None

                    distance_finger_tip_right = None
                    distance_finger_tip_left = None

                    distance_thumb_tip_right = None
                    distance_thumb_tip_left = None


                    # ------------------------------------------------
                    # Right hand
                    # ------------------------------------------------

                    if wrist_right_point is not None:

                        distance_wrist_right = float(
                            np.linalg.norm(
                                wrist_right_point - object_center
                            )
                        )


                    if finger_tip_right_point is not None:

                        distance_finger_tip_right = float(
                            np.linalg.norm(
                                finger_tip_right_point - object_center
                            )
                        )


                    if thumb_tip_right_point is not None:

                        distance_thumb_tip_right = float(
                            np.linalg.norm(
                                thumb_tip_right_point - object_center
                            )
                        )


                    # ------------------------------------------------
                    # Left hand
                    # ------------------------------------------------

                    if wrist_left_point is not None:

                        distance_wrist_left = float(
                            np.linalg.norm(
                                wrist_left_point - object_center
                            )
                        )


                    if finger_tip_left_point is not None:

                        distance_finger_tip_left = float(
                            np.linalg.norm(
                                finger_tip_left_point - object_center
                            )
                        )


                    if thumb_tip_left_point is not None:

                        distance_thumb_tip_left = float(
                            np.linalg.norm(
                                thumb_tip_left_point - object_center
                            )
                        )


                    # ------------------------------------------------
                    # Store current frame feature
                    # ------------------------------------------------

                    frame_features.append({

                        "frame": frame_id,

                        "distance_wrist_right":
                            distance_wrist_right,

                        "distance_wrist_left":
                            distance_wrist_left,

                        "distance_finger_tip_right":
                            distance_finger_tip_right,

                        "distance_finger_tip_left":
                            distance_finger_tip_left,

                        "distance_thumb_tip_right":
                            distance_thumb_tip_right,

                        "distance_thumb_tip_left":
                            distance_thumb_tip_left,

                        "conf": conf,

                        "name": name,

                        "class_id": class_id
                    })


        # Only features from this frame are returned
        return frame_features
