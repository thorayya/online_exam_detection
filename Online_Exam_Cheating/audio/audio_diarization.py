import os
import json
import argparse
import numpy as np
import librosa

from nemo.collections.asr.models import SortformerEncLabelModel


SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/diar_sortformer_4spk-v1"

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_diarization_model():
    """Load the pretrained speaker diarization model."""
    model = SortformerEncLabelModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def load_audio_files(audio_dir):
    """Load audio files as mono waveforms at 16 kHz."""
    audios = {}

    for filename in sorted(os.listdir(audio_dir)):
        filepath = os.path.join(audio_dir, filename)
        extension = os.path.splitext(filename)[1].lower()

        if not os.path.isfile(filepath) or extension not in AUDIO_EXTENSIONS:
            continue

        audio, _ = librosa.load(
            filepath,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) > 0:
            audios[filename] = audio

    return audios


def diarize_audio(model, audio):
    """Detect speaker segments with timestamps for one complete audio file."""
    result = model.diarize(
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        batch_size=1,
        verbose=False
    )

    segments = result[0] if result else []
    diarization_results = []

    for segment in segments:
        fields = segment.split()

        if len(fields) < 3:
            continue

        start = max(0.0, float(fields[0]))
        end = min(len(audio) / SAMPLE_RATE, float(fields[1]))
        speaker = fields[2]

        if end <= start:
            continue

        diarization_results.append({
            "start": start,
            "end": end,
            "speaker": speaker
        })

    return diarization_results


def process_audio_files(audio_dir, model):
    """Run diarization on every supported audio file."""
    audios = load_audio_files(audio_dir)
    results = {}

    for filename, audio in audios.items():
        try:
            segments = diarize_audio(model, audio)
            results[filename] = segments

            speakers = sorted({
                segment["speaker"] for segment in segments
            })

            print(
                f"{filename}: {len(segments)} segments, "
                f"{len(speakers)} detected speaker labels"
            )

        except Exception as error:
            print(f"Error processing {filename}: {error}")

    return results


def save_results(results, output_path):
    """Save diarization results as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Audio speaker diarization using NVIDIA Sortformer."
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing audio files."
    )
    parser.add_argument(
        "--output",
        default="sortformer_results.json",
        help="Path for the JSON output."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.audio_dir):
        raise NotADirectoryError(
            f"Audio directory does not exist: {args.audio_dir}"
        )

    model = load_diarization_model()
    results = process_audio_files(args.audio_dir, model)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
