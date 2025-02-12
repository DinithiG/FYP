import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    audio_dir = "/kaggle/input/partialspoof/NewData/train2/con_wav"
    label_file = "/kaggle/input/partialspoof/NewData/segment_labels/train_seglab_0.16.npy"
    print("Initializing dataset...")
    dataset = AudioDataset(audio_dir, label_file, subset_ratio=0.7, max_frames=100)
    print("\nPreparing data...")
    segments, labels = prepare_data(dataset)
    print("\nStarting training...")
    trained_model = train_and_evaluate(segments, labels, num_epochs=50)

    # Save the trained model
    torch.save(trained_model.state_dict(), "/kaggle/working/trained_model.pth")
    print("Trained model saved to /kaggle/working/trained_model.pth")
