import numpy as np
import soundfile as sf

input_paths = ["/content/yuqianyi_part1_enhanced.wav", "/content/yuqianyi_part2_enhanced.wav"]
output_paths = "/content/yuqianyi.wav"

audio_data = []
sample_rate = None
num_channels = None

for input_path in input_paths:
    audio, sr = sf.read(input_path)

    channels = 1 if audio.ndim == 1 else audio.shape[1]



    if sample_rate is None:
        sample_rate = sr
        num_channels = channels
        audio_data.append(audio)

    elif sample_rate != sr:
      print(
            f"Warning: Sample rate mismatch for {input_path}. "
            f"Expected {sample_rate}, got {sr}."
        )

    elif num_channels != channels:
      print(
            f"Warning: Channel count mismatch for {input_path}. "
            f"Expected {num_channels}, got {channels}."
        )

    else:
      audio_data.append(audio)

merged_audio = np.concatenate(audio_data, axis = 0)

sf.write(output_paths, merged_audio, sample_rate)
print(f"saved")
