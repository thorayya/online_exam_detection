import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from l2cs import Pipeline
import torch

class EyeTrackingGaze:

  LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
  RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
  LEFT_IRIS = [468, 469, 470, 471, 472]
  RIGHT_IRIS = [473, 474, 475, 476, 477]


  def __init__(self, model_asset_path, weights):
    self.model_asset_path = model_asset_path
    self.weights = weights


    base_options = python.BaseOptions(model_asset_path=self.model_asset_path)

    options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO)

    self.detector = vision.FaceLandmarker.create_from_options(options)


    self.gaze_pipeline = Pipeline(
    weights= self.weights,
    arch='ResNet50',
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


  def mean_normalized(self,landmark):

    left_iris_points = [ landmark[i] for i in self.LEFT_IRIS ]
    right_iris_points = [ landmark[i] for i in self.RIGHT_IRIS ]


    mean_left_iris = (sum(p.x for p in left_iris_points)/len(left_iris_points),
    sum(p.y for p in left_iris_points)/len(left_iris_points))

    mean_right_iris = (sum(p.x for p in right_iris_points)/len(right_iris_points),
      sum(p.y for p in right_iris_points)/len(right_iris_points))

    return [mean_left_iris, mean_right_iris]


  def extract(self, frame, timestamp_ms):

    frame_rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
    image_format = mp.ImageFormat.SRGB,
    data = frame_rgb)

    detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        iris_center = self.mean_normalized(landmarks)
        iris_valid = True
    else:
        iris_center = ([0.0, 0.0], [0.0, 0.0]) 
        iris_valid = False


    results = self.gaze_pipeline.step(frame)


    pitch = results.pitch
    yaw = results.yaw


    return {
    "pitch": float(pitch[0]),
    "yaw": float(yaw[0]),
    "face_confidence": float(results.scores[0]),
    "left_iris_x": iris_center[0][0],
    "left_iris_y": iris_center[0][1],
    "right_iris_x": iris_center[1][0],
    "right_iris_y": iris_center[1][1],
    "iris_valid": bool(iris_valid)
      }

