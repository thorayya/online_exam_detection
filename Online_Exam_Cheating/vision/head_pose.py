import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HeadPoseExtractor:

    def __init__(self, model_path):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

    def extract(self, frame, timestamp_ms):
        
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        detection_result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not detection_result.face_landmarks:
            return None

        face_landmarks = detection_result.face_landmarks[0]

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        image_points = np.array([
            (
                face_landmarks[4].x * frame.shape[1],
                face_landmarks[4].y * frame.shape[0]
            ),
            (
                face_landmarks[152].x * frame.shape[1],
                face_landmarks[152].y * frame.shape[0]
            ),
            (
                face_landmarks[33].x * frame.shape[1],
                face_landmarks[33].y * frame.shape[0]
            ),
            (
                face_landmarks[263].x * frame.shape[1],
                face_landmarks[263].y * frame.shape[0]
            ),
            (
                face_landmarks[61].x * frame.shape[1],
                face_landmarks[61].y * frame.shape[0]
            ),
            (
                face_landmarks[291].x * frame.shape[1],
                face_landmarks[291].y * frame.shape[0]
            )
        ], dtype=np.float64)

        height, width = frame.shape[:2]

        focal_length = width

        center = (
            width / 2,
            height / 2
        )

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix = cv2.Rodrigues(rvec)[0]

        angles = cv2.RQDecomp3x3(
            rotation_matrix
        )[0]

        return {
            "pitch": float(angles[0]),
            "yaw": float(angles[1]),
            "roll": float(angles[2]),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "rotation_matrix": rotation_matrix.tolist()
        }

