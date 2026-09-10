## Vision Module

The Vision module is responsible for extracting visual and behavioral features from webcam video as part of the online exam cheating detection system. The goal of this module is to transform raw video frames into structured, interpretable features that can later be used by a temporal deep learning model and combined with other modalities such as audio.

The module is organized into independent Python components to keep the pipeline modular, reusable, and easy to test.

### Components

**Face Detection and Verification**
The face detection component identifies the number of visible faces in each frame and verifies whether the detected face corresponds to the registered user. It can flag events such as the absence of a face, multiple faces, or a face mismatch.

**Eye and Gaze Tracking**
The eye-tracking component uses MediaPipe Face Landmarker and L2CS-Net to extract gaze-related information. It estimates head/gaze orientation through pitch and yaw values and calculates normalized iris positions for both eyes.

**Head Pose Estimation**
Head pose information is extracted from facial landmarks and transformation information to represent the user's orientation and possible changes in attention direction over time.

**Object Detection**
The object detection component identifies potentially relevant objects such as mobile phones, laptops, books, and earphones. YOLO-based detection is combined with a custom earphone detection model.

**Human Pose and Tracking**
MediaPipe Pose Landmarker is used to extract body keypoints including wrists, elbows, and shoulders. Object tracking is performed to maintain temporal information about detected objects across frames.

**Spatial Feature Extraction**
The module derives higher-level behavioral features from the detected keypoints and objects. These include normalized distances between hands/elbows and nearby objects, as well as elbow joint angles.

### Output

The extracted information is organized at the frame level and subsequently represented as temporal sequences. These sequences will be converted into tensors and used as the visual input of the multimodal cheating-detection architecture.

The modular design allows each component to be independently tested, improved, or replaced without modifying the entire pipeline.
