import os
import random
import shutil

# Define the path to your dataset
data_path = 'G:/FCAI/GP/dataset'

# Define the ratios for the data split
train_ratio = 0.55
val_ratio = 0.20
test_ratio = 0.25

# Define the paths to the train, validation, and test directories
train_dir = os.path.join(data_path, 'train')
val_dir = os.path.join(data_path, 'val')
test_dir = os.path.join(data_path, 'test')


# Loop through each class directory
for class_name in os.listdir(data_path):
    if not os.path.isdir(os.path.join(data_path, class_name)):
        continue

    class_dir = os.path.join(data_path, class_name)

    # Get a list of audio files in the class directory
    audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

    # Shuffle the audio files
    random.shuffle(audio_files)

    # Calculate the number of audio files for each split
    num_train = int(len(audio_files) * train_ratio)
    num_val = int(len(audio_files) * val_ratio)
    num_test = len(audio_files) - num_train - num_val

    # Create the directories for the current class
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Split the audio files into train, validation, and test sets
    train_files = audio_files[:num_train]
    val_files = audio_files[num_train:num_train + num_val]
    test_files = audio_files[num_train + num_val:]

    # Copy the audio files to the appropriate directories
    for file_name in train_files:
        shutil.copy(os.path.join(class_dir, file_name), os.path.join(train_dir, class_name, file_name))

    for file_name in val_files:
        shutil.copy(os.path.join(class_dir, file_name), os.path.join(val_dir, class_name, file_name))

    for file_name in test_files:
        shutil.copy(os.path.join(class_dir, file_name), os.path.join(test_dir, class_name, file_name))
