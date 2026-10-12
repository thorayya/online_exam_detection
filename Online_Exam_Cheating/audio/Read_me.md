# Audio Processing Pipeline

This repository contains audio preprocessing and speaker diarization tools developed as part of a multimodal online examination integrity research project.

The pipeline includes audio denoising, audio splitting and merging, and speaker diarization. These components prepare audio recordings for subsequent analysis and multimodal fusion.

## 1. Audio Denoising with DeepFilterNet3

A Python-based tool for batch noise reduction using DeepFilterNet3.

### Features

- **Batch Processing:** Processes WAV files organized into subject-specific directories.
- **Deep Learning-Based Enhancement:** Uses DeepFilterNet3 to reduce background noise.
- **Efficient Initialization:** Loads the enhancement model once per processing session.
- **Organized Output:** Prefixes output filenames with their subject folder names.
- **Skip Existing Files:** Avoids reprocessing previously saved recordings.
- **Error Handling:** Continues processing when an individual recording fails.

### Requirements

- Python 3.11
- DeepFilterNet 0.5.6
- Compatible PyTorch and TorchAudio versions

### Installation

```bash
pip install deepfilternet==0.5.6
```

### Dataset Structure

The input directory should contain subject-specific folders:

```text
dataset/
├── subject1/
│   ├── recording1.wav
│   └── recording2.wav
├── subject2/
│   ├── recording1.wav
│   └── recording2.wav
└── subject3/
    └── recording1.wav
```

### Configuration and Usage

Update `dataset_path` and `output_dir` in `audio_denoising.py` to match your environment.

Run the script:

```bash
python audio_denoising.py
```

Enhanced recordings are saved to the configured output directory using the following naming convention:

```text
<subject_name>_<original_filename>.wav
```

Existing output files are skipped.

---

## 2. Splitting Large Audio Files

The `split_audio.py` script divides an audio file into two approximately equal parts and saves them as separate WAV files.

### Usage

Configure the input audio path and the output paths for both segments in the script, then run:

```bash
python split_audio.py
```

The script preserves the original sample rate. It loads the audio into memory, so very large recordings may require substantial RAM.

---

## 3. Merging Audio Segments

The `merge_audio.py` script concatenates multiple WAV segments into a single audio file. It can be used to reconstruct recordings that were previously split into smaller parts, including denoised recordings.

### Features

- Concatenates segments in the specified order.
- Validates sample rates and channel counts.
- Saves the merged recording as a WAV file.
- Skips incompatible segments rather than automatically resampling them or converting their channel layouts.

### Requirements

- Python
- NumPy
- SoundFile

Install the dependencies:

```bash
pip install numpy soundfile
```

### Usage

1. Set `input_paths` to the audio segments in the desired order.
2. Set `output_path` to the destination WAV file.
3. Run the script:

```bash
python merge_audio.py
```

All segments must have matching sample rates and compatible channel configurations. Ensure that the input paths are ordered correctly to preserve the intended recording sequence.

---

## 4. Speaker Diarization with NVIDIA Sortformer

The `audio_diarization.py` script identifies speaker segments in audio recordings using NVIDIA's pretrained Sortformer model.

### Features

- Supports WAV, FLAC, MP3, OGG, and M4A input files.
- Converts recordings to mono audio at 16 kHz.
- Predicts speaker labels and their corresponding time intervals.
- Preserves timestamps relative to the original audio recording.
- Exports results to JSON for downstream multimodal fusion.

### Requirements

- A Python environment compatible with the installed NeMo release
- librosa
- NumPy
- `nemo_toolkit[asr]`

### Installation

```bash
pip install librosa numpy "nemo_toolkit[asr]"
```

### Usage

Run the script with the directory containing the audio recordings and the desired output path:

```bash
python audio_diarization.py \
  --audio-dir "path/to/audios" \
  --output "sortformer_results.json"
```

The script loads the pretrained model:

```text
nvidia/diar_sortformer_4spk-v1
```

Internet access may be required on the first run to download the model weights.

### Output Format

The JSON file contains the predicted speaker segments for each recording:

```json
{
  "recording.wav": [
    {
      "start": 0.32,
      "end": 2.15,
      "speaker": "speaker_0"
    },
    {
      "start": 2.40,
      "end": 3.10,
      "speaker": "speaker_1"
    }
  ]
}
```

Timestamps are measured in seconds relative to the original audio file. Speaker labels represent model predictions and should not be assumed to identify the true number of people without validation.

---

## Research Context

These tools form part of the audio processing pipeline for a multimodal online examination integrity research project.

Audio denoising improves recording quality, splitting and merging support audio file management, and speaker diarization provides speaker-related temporal information. The resulting data can be used in subsequent audio analysis and multimodal fusion with vision and mobile-derived features.

Speaker count or the presence of multiple speakers alone does not establish suspicious behavior or cheating. Audio-derived evidence should be interpreted alongside other modalities and evaluated using appropriate validation procedures.

## Reproducibility and Data Usage

- Document environment-specific settings and compatible dependency versions.
- Verify model and library compatibility before running the scripts.
- Keep large recordings and generated intermediate files outside the repository unless they are explicitly needed and permitted for publication.
- Share datasets only when their licenses and applicable privacy requirements permit redistribution.

## License

Add a repository license specifying the terms under which the code may be used, modified, and distributed. Check the licenses of third-party models, libraries, and datasets separately.


