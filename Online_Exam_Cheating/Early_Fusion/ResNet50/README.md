## ResNet-50 Feature Extraction for Online Exam Cheating Detection

### Overview
This module extracts visual features from video recordings in the Online Exam Proctoring (OEP) dataset using a pretrained ResNet-50 model. The extracted features can be used as input for downstream machine learning and deep learning models for online exam behavior analysis.

### Methodology
- **Video Selection:** Selects a video from the dataset based on its subject folder and sorted video order.
- **Frame Sampling:** Samples one frame every 25 original video frames.
- **Preprocessing:** Converts frames from BGR to RGB and preprocesses them using the ResNet-50 image processor.
- **Feature Extraction:** Uses a pretrained `microsoft/resnet-50` model from Hugging Face. The model is frozen and used in evaluation mode to extract visual feature maps.
- **Feature Representation:** Flattens the final feature maps from `(B, 2048, 7, 7)` into feature vectors of 100,352 dimensions per frame.
- **Temporal Labeling:** Assigns normal, suspicious, and cheating labels according to predefined time intervals in seconds.
- **Batch Processing:** Processes frames in batches to reduce memory usage.
- **Output Storage:** Saves extracted features as a NumPy `.npy` file and frame metadata and labels as a JSON file.

### Labels
Each sampled frame receives three binary labels:

- `normal`: The frame falls outside the predefined suspicious and cheating intervals.
- `suspicious`: The frame timestamp falls within an interval assigned to a suspicious class.
- `cheating`: The frame timestamp falls within an interval assigned to a cheating class.

The class IDs are grouped as follows:

- Suspicious: `1`, `3`, `5`, `6`
- Cheating: `2`, `4`

These labels are based on predefined annotation intervals and do not represent independent predictions made by ResNet-50.

### Output Files
- `<video_name>_features.npy`: Extracted visual feature vectors.
- `<video_name>_labels.json`: Frame numbers, timestamps, and binary labels.

### Technologies
Python, PyTorch, Hugging Face Transformers, ResNet-50, OpenCV, NumPy, and JSON.

### Intended Use
The extracted features are intended for subsequent experiments in video-based anomaly detection and online examination integrity analysis. They can be used to train or evaluate downstream classification models.

### Notes
- Timestamps are measured in seconds.
- The script processes one selected video per execution.
- ResNet-50 is used as a frozen feature extractor; no model fine-tuning is performed.
- The extracted features and labels must remain aligned for subsequent training and evaluation.
- The quality of the labels depends on the accuracy of the predefined annotation intervals.

