import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from deepface import DeepFace

class FaceDetector:

    def __init__(self, model_path):


        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.FaceDetector.create_from_options(
            options
        )

    def face_detection(self, timestamp_ms, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        detection_result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_count = len(
            detection_result.detections
        )

        deepface_result = None


        return {
            "face_count": face_count,
        }




