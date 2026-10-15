
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
    """Load complete audio files as mono waveforms at 16 kHz."""
    audios = {}

    for filename in sorted(os.listdir(audio_dir)):
        filepath = os.path.join(audio_dir, filename)
        extension = os.path.splitext(filename)[1].lower()

        if not os.path.isfile(filepath):
            continue

        if extension not in AUDIO_EXTENSIONS:
            continue

        audio, sr = librosa.load(
            filepath,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) > 0:
            audios[filename] = audio

        print(
            f"Loaded {filename}: "
            f"{len(audio) / SAMPLE_RATE:.2f} seconds"
        )

    return audios


def diarize_audio(model, audio):
    """
    Run Sortformer on the complete audio file.
    Returned timestamps are interpreted relative to the original file.
    """
    result = model.diarize(
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        batch_size=1,
        verbose=False
    )

    segments = result[0] if result else []
    duration = len(audio) / SAMPLE_RATE
    diarization_results = []

    for segment in segments:
        fields = str(segment).split()

        if len(fields) < 3:
            continue

        try:
            start = float(fields[0])
            end = float(fields[1])
            speaker = fields[2]
        except (ValueError, IndexError):
            continue

        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))

        if end <= start:
            continue

        diarization_results.append({
            "start": start,
            "end": end,
            "speaker": speaker
        })

    diarization_results.sort(key=lambda item: item["start"])

    return diarization_results


def process_audio_files(audio_dir, model):
    """Diarize every supported audio file without merging speech intervals."""
    audios = load_audio_files(audio_dir)
    results = {}
    durations = {}

    for filename, audio in audios.items():
        durations[filename] = len(audio) / SAMPLE_RATE

        try:
            segments = diarize_audio(model, audio)
            results[filename] = segments

            speakers = sorted({
                segment["speaker"] for segment in segments
            })

            print(
                f"{filename}: "
                f"{len(segments)} segments, "
                f"{len(speakers)} speaker labels, "
                f"duration={durations[filename]:.2f}s"
            )

        except Exception as error:
            print(f"Error processing {filename}: {error}")

    return results, durations


def save_results(results, durations, output_path):
    """Save segments and original audio durations as JSON."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "sample_rate": SAMPLE_RATE,
        "durations": durations,
        "segments": results
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization using NVIDIA Sortformer."
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

    results, durations = process_audio_files(
        args.audio_dir,
        model
    )

    save_results(
        results,
        durations,
        args.output
    )


if __name__ == "__main__":
    main()

