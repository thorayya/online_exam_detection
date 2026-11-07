
## Mobile Self-Attention Preprocessing

This script preprocesses mobile-related features for the mobile branch of a multimodal online exam integrity detection framework.

### Main Steps

1. Load mobile detection data from PKL files.
2. Extract wrist, finger-tip, thumb-tip distances, confidence scores, and class IDs.
3. Group extracted features into video-level representations.
4. Split videos into training, validation, and test sets.
5. Generate temporal windows and corresponding masks.
6. Normalize model input features using training-set statistics.
7. Save processed data, masks, feature names, and temporal metadata.

### Feature Configuration

- Raw feature dimension: 8
- Model input dimension: 7
- Window size: 10 seconds
- Stride: 8 seconds
- Sampling interval: 25 frames
- Source frame rate: 25 FPS

The `class_id` feature is retained in the raw data but excluded from the seven-dimensional model input.

### Temporal Metadata

Each window stores:

- Video ID and source filename
- Window ID and video index
- Start and end sample indices
- Start and end timestamps
- Valid sequence length
- Data split

### Output

The script saves the processed dataset as:

`mobile_self_attention_data.pt`

The output contains raw and normalized windows, masks, per-video features, normalization statistics, split indices, and temporal metadata.

### Requirements

- Python
- PyTorch
- Google Colab
- Google Drive

### Dataset

The input PKL files are expected to contain a list of timestep records, each with an `objects` field.

Dataset files are not included in this repository.
