
import soundfile as sf

input_path = "/content/makiluke.wav"

audio_info = sf.info(input_path)
audio_seconds = audio_info.duration
first_section = audio_seconds / 2

audio, sr = sf.read(input_path)
mid = int(first_section * sr)

part1_path = "/content/makiluke_part1.wav"
part2_path = "/content/makiluke_part2.wav"

sf.write(part1_path, audio[:mid], sr)
sf.write(part2_path, audio[mid:], sr)

print("First part:", part1_path)
print("Second part:", part2_path)
