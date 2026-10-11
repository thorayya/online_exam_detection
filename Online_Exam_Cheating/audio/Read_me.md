### **Audio Denoising with DeepFilterNet3**



A Python-based audio preprocessing tool for batch noise reduction using DeepFilterNet3, developed as part of a multimodal online exam cheating detection research project.



###### Overview



This project applies DeepFilterNet3 to WAV audio recordings organized into subject-specific folders. The enhanced recordings are saved to a separate output directory for subsequent audio analysis.



###### **Features**



Batch Processing: Processes WAV files from multiple subject directories.



DeepFilterNet3: Uses a deep learning-based audio enhancement model.



Efficient Initialization: Loads the model once per processing session.



Organized Output: Prefixes output filenames with their corresponding subject folder names.



Skip Existing Files: Avoids reprocessing files that have already been saved.



Error Handling: Continues processing other recordings when an individual file encounters an error.



###### **Requirements**



Python 3.11



DeepFilterNet 0.5.6



Compatible PyTorch and TorchAudio versions



###### **Installation**



Install DeepFilterNet in a compatible Python environment:



pip install deepfilternet==0.5.6



###### **Dataset Structure**



The input dataset should follow this structure:



dataset/

├── subject1/

│   ├── recording1.wav

│   └── recording2.wav

├── subject2/

│   ├── recording1.wav

│   └── recording2.wav

└── subject3/

&#x20;   └── recording1.wav



###### **Configuration**



Before running the script, update the following paths in audio\_denoising.py to match your environment:



dataset\_path: Directory containing the subject folders and WAV recordings.



output\_dir: Directory where the enhanced recordings will be saved.



###### **Usage**



Run the script from an environment where the required dependencies are installed:



&#x09;python audio\_denoising.py



The enhanced recordings will be saved to the specified output directory.



###### **Output**



Output filenames follow this format:



<subject\_name>\_<original\_filename>.wav



Existing output files are skipped to avoid unnecessary processing.



###### **Research Context**



This tool serves as an audio preprocessing component in a multimodal online exam cheating detection research project. The denoised recordings can be used in subsequent audio analysis and multimodal learning experiments.



Reproducibility



For reproducible results, use compatible dependency versions and document any environment-specific configuration changes.



###### **License**



A license can be added to specify the terms under which this code may be used, modified, and distributed. Ensure that any dataset or audio recordings are shared only when their licenses permit redistribution.



### Splitting Large Audio Files

If an audio file is too large to process at once, this script splits it into two approximately equal parts and saves them as separate WAV files.

The script preserves the original sample rate and divides the audio at its midpoint.


### Merging Audio Segments

This script merges multiple WAV audio segments into a single audio file. It is designed to reconstruct audio recordings that were previously split into smaller parts, including enhanced recordings generated during audio preprocessing.

**Features:**
- **Segment Concatenation:** Combines audio segments in the specified order.
- **Sample Rate Validation:** Checks whether the segments have matching sample rates.
- **Channel Validation:** Checks whether the segments have matching numbers of audio channels.
- **WAV Output:** Saves the merged recording as a WAV file.

**Requirements:**
- Python
- NumPy
- `soundfile`

Install the required packages:

```bash
pip install numpy soundfile
```

**Usage:**

1. Set `input_paths` to the paths of the audio segments in the desired order.
2. Set `output_paths` to the destination path for the merged WAV file.
3. Run the script:

```bash
python merge_audio.py
```

**Output:**

The script concatenates the compatible audio segments and saves the resulting recording to the specified output path.

**Important Notes:**
- All segments included in the merged output must have matching sample rates and compatible channel configurations.
- If a segment has a different sample rate or number of channels, the script prints a warning and excludes that segment from the merged output.
- The script does not automatically resample audio or convert between mono and stereo.
- The segments are concatenated in the order specified in `input_paths`; ensure they are listed correctly to preserve the original recording sequence.

**Research Context:**

This script is part of the audio preprocessing pipeline for a multimodal online exam cheating detection research project. It supports the reconstruction of complete audio recordings from smaller segments before subsequent audio analysis and multimodal learning experiments.

**Requirements:**
- Python
- `soundfile`

Install the required package:

```bash
pip install soundfile
```

**Usage:**

1. Set `input_path` to the location of your audio file.
2. Set `part1_path` and `part2_path` to the desired output locations.
3. Run the script:

```bash
python split_audio.py
```

**Note:** The script loads the entire audio file into memory. For very large files or systems with limited RAM, a memory-efficient approach may be necessary.


