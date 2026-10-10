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

