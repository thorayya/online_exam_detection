

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

earphone_model_path = "/content/drive/MyDrive/Online_Exam_Cheating/best.pt"

class ObjectDetection:
  def __init__(self, model_path, yolo_model_path="yolov8.pt", earphone_model_path=earphone_model_path):

    self.model_path = model_path
    self.yolo_model_path = yolo_model_path
    self.earphone_model_path = earphone_model_path

    self.BaseOptions = mp.tasks.BaseOptions
    self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
    self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    self.PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    self.VisionRunningMode = mp.tasks.vision.RunningMode

    self.classes = ["cell phone", "book", "laptop", "earphone"]
    self.CLASS_IDS = {"cell phone": 0, "book": 1, "laptop": 2, "earphone": 4}

    self.model = None
    self.model_earphone = None
    self.landmarker = None

    self.model, self.model_earphone, self.landmarker = self.load_models()


  def load_models(self):
    self.model = YOLO(self.yolo_model_path)
    self.model_earphone = YOLO(self.earphone_model_path)

    self.model.to("cuda")
    self.model_earphone.to("cuda")

    options = self.PoseLandmarkerOptions(
      base_options=self.BaseOptions(model_asset_path=self.model_path),
      running_mode=self.VisionRunningMode.VIDEO)


    self.landmarker = self.PoseLandmarker.create_from_options(options)

    return self.model, self.model_earphone, self.landmarker


  def object_detection(self,frame):

    self.objects_boxes = []
    self.dets = []

    results_main = self.model(frame, verbose=False)
    results_earphone = self.model_earphone(frame, verbose=False)

    model_results = ([
        (results_main, self.model),
        (results_earphone, self.model_earphone)
    ])

    for results, model in model_results:
      for result in results:
        boxes = result.boxes

        for box in boxes:
          x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()  # Get the coordinates
          conf = box.conf[0].cpu().numpy() # Confidence score
          cls = int(box.cls[0].cpu().numpy())# Class ID

          name = model.names[cls]

          if name in self.classes:

            global_cls = self.CLASS_IDS[name]

            self.dets.append([x1,y1,x2,y2,conf,global_cls])

            center_x = (x1 + x2)/ 2
            center_y =(y1 + y2)/2

            self.objects_boxes.append((center_x,center_y))

    return self.objects_boxes, self.dets

  def pose_detection(self,frame,timestamp_ms, w, h):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
    )
    
    result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks:
      return None

    else:

      self.landmark = result.pose_landmarks[0]


      right_wrist = self.landmark[16]
      left_wrist = self.landmark[15]

      left_shoulder = self.landmark[11]
      right_shoulder = self.landmark[12]

      right_elbow = self.landmark[14]
      left_elbow = self.landmark[13]


      x_right_wrist = int(right_wrist.x * w)
      y_right_wrist = int(right_wrist.y * h)

      x_left_wrist = int(left_wrist.x * w)
      y_left_wrist = int(left_wrist.y * h)

      x_left_shoulder = int(left_shoulder.x * w)
      y_left_shoulder = int(left_shoulder.y * h)

      x_right_shoulder = int(right_shoulder.x * w)
      y_right_shoulder = int(right_shoulder.y * h)

      x_right_elbow = int(right_elbow.x * w)
      y_right_elbow = int(right_elbow.y * h)

      x_left_elbow = int(left_elbow.x * w)
      y_left_elbow = int(left_elbow.y * h)

      position={
            "right_wrist": (x_right_wrist, y_right_wrist),
            "left_wrist": (x_left_wrist, y_left_wrist),
            "left_shoulder": (x_left_shoulder, y_left_shoulder),
            "right_shoulder": (x_right_shoulder, y_right_shoulder),
            "right_elbow": (x_right_elbow, y_right_elbow),
            "left_elbow": (x_left_elbow, y_left_elbow)
        }

    return position


  def build_features(self, pose, objects_boxes):

    if pose is None or len(objects_boxes) == 0:
      return None

    else:
      distances = []
      for box in objects_boxes:
        d1 = np.linalg.norm(np.array(pose["left_wrist"]) - np.array(box))
        d2 = np.linalg.norm(np.array(pose["right_wrist"]) - np.array(box))

        distances.append(min(d1,d2))

      nearest_object = np.argmin(distances)
      obj = objects_boxes[nearest_object]

      # ---- normalize ------
      shoulder_width = (np.linalg.norm(np.array(pose["left_shoulder"]) - np.array(pose["right_shoulder"]))) 

      left_wrist_distance_to_object = (np.linalg.norm(np.array(pose["left_wrist"]) - np.array(obj))) / (shoulder_width + 1e-6)
      right_wrist_distance_to_object = (np.linalg.norm(np.array(pose["right_wrist"]) - np.array(obj))) / (shoulder_width + 1e-6)

      left_elbow_distance_to_object = (np.linalg.norm(np.array(pose["left_elbow"])- np.array(obj))) / (shoulder_width + 1e-6)
      right_elbow_distance_to_object = (np.linalg.norm(np.array(pose["right_elbow"]) - np.array(obj))) / (shoulder_width + 1e-6)

      return (
              left_wrist_distance_to_object,
              right_wrist_distance_to_object,
              left_elbow_distance_to_object,
              right_elbow_distance_to_object
              )


  def angle(self, pose_shoulder, pose_elbow, pose_wrist):

    pose_shoulder = np.array(pose_shoulder)
    pose_elbow = np.array(pose_elbow)
    pose_wrist = np.array(pose_wrist)

    elbow_shoulder = pose_shoulder - pose_elbow
    wrist_elbow = pose_wrist - pose_elbow

    cosine = np.dot(elbow_shoulder, wrist_elbow) / (np.linalg.norm(elbow_shoulder) * np.linalg.norm(wrist_elbow) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

