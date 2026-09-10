
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

class HeadPoseExtractor:

  NOSE = 1
  CHIN = 152
  LEFT_EYE = 33
  RIGHT_EYE = 263
  MOUTH_LEFT = 61
  MOUTH_RIGHT = 291

  def __init__(self, model_path):

    self.model_path = model_path
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode


    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO)

    self.detector = vision.FaceLandmarker.create_from_options(options)

    self.model_points = np.array([
    [0, 0, 0],
    [0, -330, -65],
    [-225, 170, -135],
    [225, 170, -135],
    [-150, -150, -125],
    [150, -150, -125]
      ], dtype=np.float64)

    self.dist_coeffs = np.zeros((4,1))


# 'face_landmarker_v2_with_blendshapes.task'



  def focal_length(self, h, w):

    focal_length = 1000

    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])

    return camera_matrix


  def extract(self, frame, timestamp):

    h, w = frame.shape[:2]

    camera_matrix = self.focal_length(h, w)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = self.detector.detect_for_video(mp_image, timestamp)


    if detection_result.face_landmarks:

      landmark = detection_result.face_landmarks[0]

      nose = landmark[self.NOSE]
      chin = landmark[self.CHIN]
      left_eye = landmark[self.LEFT_EYE]
      right_eye = landmark[self.RIGHT_EYE]
      mouth_left = landmark[self.MOUTH_LEFT]
      mouth_right = landmark[self.MOUTH_RIGHT]


      image_points = np.array([
                              [nose.x*w, nose.y*h],
                              [chin.x*w, chin.y*h],
                              [left_eye.x*w, left_eye.y*h],
                              [right_eye.x*w, right_eye.y*h],
                              [mouth_left.x*w, mouth_left.y*h],
                              [mouth_right.x*w, mouth_right.y*h]
                              ],
                              dtype = np.float64)


      success, rvec, tvec = cv2.solvePnP(
      self.model_points,
      image_points,
      camera_matrix,
      self.dist_coeffs,
      flags=cv2.SOLVEPNP_ITERATIVE
      )

      if success:

        rotation_matrix = cv2.Rodrigues(rvec)[0]
        angles, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]


        projected_points,_ =cv2.projectPoints(
            self.model_points,
            rvec,
            tvec,
            camera_matrix,
            self.dist_coeffs
        )

        reprojection_error = np.linalg.norm(image_points - projected_points.squeeze(),
                               axis=1).mean()

        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'timestamp': timestamp,
            'reprojection_error': float(reprojection_error)
        }

      return {
          'pitch': None,
          'yaw': None,
          'roll': None,
          'timestamp': timestamp,
          'reprojection_error': None
        }

    return {
        'pitch': None,
        'yaw': None,
        'roll': None,
        'timestamp': timestamp,
        'reprojection_error': None
    }
