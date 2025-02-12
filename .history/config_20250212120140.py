import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
import librosa
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Example usage of some libraries
print("Libraries imported successfully!")