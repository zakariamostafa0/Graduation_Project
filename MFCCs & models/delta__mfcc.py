import librosa
import os
import math
import json
import numpy as np

DATASET_PATH = "G:/FCAI/GP/Improved_dataset"
JSON_OUT_DIR = "G:/FCAI/Python Projects/Stuttering-Classification/"

# Define the names of the three folders
TRAIN_DIR = "train"
TEST_DIR = "test"
VAL_DIR = "val"

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=14, n_fft=2048, hop_length=512, num_segments=1):
    # dictionary to store data
    data = {
        "mapping": [],
        "features": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    print("mfcc shape: ", mfcc.shape)

                    # extract delta MFCCs
                    delta_mfcc = librosa.feature.delta(mfcc)
                    print("delta shape: ", delta_mfcc.shape)

                    # extract double-delta MFCCs
                    double_delta_mfcc = librosa.feature.delta(mfcc, order=2)
                    print("double shape: ", double_delta_mfcc.shape)

                    # concatenate the three feature sets
                    #features = np.concatenate((mfcc, delta_mfcc, double_delta_mfcc), axis=0)
                    features = mfcc + delta_mfcc + double_delta_mfcc
                    print("features shape before T: ", features.shape)
                    features = features.T
                    print("features shape: ", features.shape)

                    # store mfcc for segment if it has the expected length
                    if len(features) == expected_num_mfcc_vectors_per_segment:
                        data["features"].append(features.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    # Loop over the three folders and call the `save_mfcc` function for each
    for dir_name in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        # Construct the full path to the current directory
        dir_path = os.path.join(DATASET_PATH, dir_name)

        # Construct the full path to the output JSON file
        json_path = os.path.join(JSON_OUT_DIR, f"{dir_name}_delta14sumAug.json")
        # Call the `save_mfcc` function for the current directory
        save_mfcc(dir_path, json_path, num_segments=1)
