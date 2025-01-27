import numpy as np
import os
import librosa
import random

# Load the segment labels
train_seglab = np.load("/kaggle/input/partialspoof/NewData/segment_labels/train_seglab_0.16.npy", allow_pickle=True).item()

# List of audio files from the train set
train_audio_files = list(train_seglab.keys())

# Select 5% of the dataset
subset_files = random.sample(train_audio_files, int(0.05 * len(train_audio_files)))

# Example of how the audio files and their corresponding labels look
print(subset_files[:5])
