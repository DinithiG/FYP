# 🎧 WavSpoof: Partially Spoofed Audio Detection for Speaker Verification

WavSpoof is a deep learning-based system designed to detect **partially spoofed** audio segments in speaker verification pipelines. Unlike traditional systems that only detect fully spoofed utterances, WavSpoof identifies **tampered regions** within real utterances — offering fine-grained protection against modern voice manipulation attacks.

## 🚀 Features

- 🧠 **Hybrid Deep Learning Model**: Combines LCNN with attention and residual blocks for robust spoof detection.
- 🔍 **Rich Audio Feature Extraction**:
  - MFCC
  - CQT
  - Spectral Contrast
  - Tonnetz
  - Mel-spectrogram delta
  - HuBERT embeddings
- 🎯 **Segment-Level Classification**: Detects spoofed vs. genuine regions in each utterance.
- 🧪 **Benchmarking**: Compared against the BAM (Boundary Aware Model) using a common subset of PartialSpoof.
- 🌐 **Web App**: Deployed via Next.js frontend for interactive demo and evaluation.

## 📁 Dataset

### PartialSpoof v1.2
- Derived from [ASVspoof 2019]
- Includes segment-level annotations with multiple temporal resolutions
- Voice Activity Detection (VAD) supported

## 🏗️ Model Architecture

- Concatenated multi-view feature input
- Residual Blocks with BatchNorm + ReLU
- Attention Mechanisms
- Max Pooling, Dropout
- Fully Connected Layers
- Output Layer for binary classification (Real vs. Spoofed segment)

## 🧪 Evaluation Metrics

- Accuracy
- F1 Score
- Equal Error Rate (EER)
- min-tDCF
- ROC/AUC

## 🛠️ Tech Stack

- Python (PyTorch, Librosa, NumPy, Scikit-learn)
- Feature extraction pipeline with audio preprocessing
- Next.js for web deployment
- Matplotlib & Seaborn for evaluation plots

