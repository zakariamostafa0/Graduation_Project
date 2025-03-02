import audiomentations as A
import librosa
import soundfile as sf
import os

parent_dir = "G:/FCAI/GP/Improved_dataset"

dir_name = "train"  # replace with the name of the folder you want to process

# Construct the full path to the directory
curr_dir = os.path.join(parent_dir, dir_name)

# Check if the path is a directory
if os.path.isdir(curr_dir):
    print(f"Processing directory: {curr_dir}")

    # List all subdirectories in the directory
    sub_dir_list = os.listdir(curr_dir)

    # Loop through each subdirectory
    for sub_dir_name in sub_dir_list:
        # Construct the full path to the current subdirectory
        sub_dir = os.path.join(curr_dir, sub_dir_name)

        # Check if the current path is a directory
        if os.path.isdir(sub_dir):
            print(f"Processing subdirectory: {sub_dir}")
            # Add your code for processing the files in this subdirectory here
            for filename in os.listdir(sub_dir):
                file_path = os.path.join(sub_dir, filename)
                # Do something with the file
                # Load an audio file
                audio, sr = librosa.load(file_path)

                # Create an augmentation pipeline
                augmentations = A.Compose([
                    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    A.PitchShift(min_semitones=-4, max_semitones=4, p=1),
                    A.TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
                ])

                # Apply the augmentation pipeline to the audio file
                augmented_audio = augmentations(audio, sample_rate=sr)

                # Save the augmented audio file with a new name
                filename_no_ext, ext = os.path.splitext(filename)
                new_filename = f"{filename_no_ext}_augmented{ext}"
                new_file_path = os.path.join(sub_dir, new_filename)
                sf.write(new_file_path, augmented_audio, sr)

