import os
from df.enhance import enhance, init_df, load_audio, save_audio


# 1. Define input and output directories
dataset_path = "/content/oep_dataset/OEP database"
output_dir = "/content/enhanced_audio"

os.makedirs(output_dir, exist_ok=True)


# 2. Collect WAV files for each subject
audio_paths = []

for subject in sorted(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    audio_files = sorted(
        filename
        for filename in os.listdir(subject_path)
        if filename.lower().endswith(".wav")
    )

    audio_paths.append({
        "subject": subject,
        "audio": audio_files,
    })

print(f"Found {len(audio_paths)} subject folders.")


# 3. Initialize DeepFilterNet once
model, df_state, _ = init_df()
sample_rate = df_state.sr()


# 4. Denoise and save audio files
for item in audio_paths:
    subject = item["subject"]
    subject_path = os.path.join(dataset_path, subject)

    for filename in item["audio"]:
        input_path = os.path.join(subject_path, filename)

        output_filename = f"{subject}_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Skip files that have already been processed
        if os.path.exists(output_path):
            print(f"Already exists, skipping: {output_filename}")
            continue

        try:
            audio, _ = load_audio(
                input_path,
                sr=sample_rate,
            )

            enhanced = enhance(model, df_state, audio)

            save_audio(
                output_path,
                enhanced,
                sample_rate,
            )

            print(f"Saved: {output_filename}")

        except Exception as error:
            print(f"Error processing {input_path}: {error}")


print("Finished processing audio files.")


