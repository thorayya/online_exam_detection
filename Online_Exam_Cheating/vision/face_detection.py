import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from deepface import DeepFace


class FaceDetector:
  def __init__(self, model_path, image_user_path):
    self.model_path = model_path  
    self.image_user_path = image_user_path
    self.base_options = python.BaseOptions(model_asset_path=model_path)
    self.options = vision.FaceDetectorOptions(base_options=self.base_options)

    self.detector = vision.FaceDetector.create_from_options(self.options)
    self.cheating_events = []

    self.image_user = cv2.imread(image_user_path)

  def face_detection(self, timestamp_ms, frame):

    self.cheating_detected = False
    reason = None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
    result = DeepFace.verify(img1_path=frame, img2_path=self.image_user)

    face_count = len(detection_result.detections)

    if face_count == 0:
      self.cheating_detected = True
      reason = "no_face"
    elif face_count > 1:
      self.cheating_detected = True
      reason = "multiple_faces"
    else:
      try:
        if not result["verified"]:
          self.cheating_detected = True
          reason = "face_mismatch"

      except Exception:
        self.cheating_detected = True
        reason = "verification_error"


    if self.cheating_detected:
      event={
          'time': time.strftime('%Y-%m-%d %H:%M:%S'),
          'reason': reason
      }

      self.cheating_events.append(event)
