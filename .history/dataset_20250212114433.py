class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_file, subset_ratio=0.02, max_frames=100):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        self.labels_dict = np.load(label_file, allow_pickle=True).item()
        self.audio_files = [f for f in self.audio_files if os.path.splitext(f)[0] in self.labels_dict]
        num_files = int(len(self.audio_files) * subset_ratio)
        self.audio_files = self.audio_files[:num_files]
        self.audio_files = [os.path.join(audio_dir, f) for f in self.audio_files]

        # MFCC Transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "center": True,
                "normalized": True,
            },
        )
        self.max_frames = max_frames
        print(f"Dataset initialized with {len(self.audio_files)} files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            segment_labels = self.labels_dict[file_id]
            num_segments = len(segment_labels)
            segment_labels = torch.tensor([float(label) for label in segment_labels], dtype=torch.float32)

            waveform = waveform.squeeze(0)

            # Compute MFCC
            mfcc = self.mfcc_transform(waveform)

            # Compute CQT using librosa
            cqt = librosa.cqt(waveform.numpy(), sr=16000, n_bins=84, bins_per_octave=12, hop_length=160)
            cqt = np.abs(cqt)  # Take the magnitude of the CQT
            cqt = torch.tensor(cqt, dtype=torch.float32)

            # Compute Chroma Features
            chroma = librosa.feature.chroma_stft(S=np.abs(librosa.stft(waveform.numpy())), sr=16000)
            chroma = torch.tensor(chroma, dtype=torch.float32)

            # Compute Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(waveform.numpy())), sr=16000)
            spectral_contrast = torch.tensor(spectral_contrast, dtype=torch.float32)

            # Compute Tonnetz
            tonnetz = librosa.feature.tonnetz(y=waveform.numpy(), sr=16000)
            tonnetz = torch.tensor(tonnetz, dtype=torch.float32)

            # Compute Mel Spectrogram Deltas
            mel_spectrogram = librosa.feature.melspectrogram(y=waveform.numpy(), sr=16000, n_fft=400, hop_length=160)
            mel_spectrogram_delta = librosa.feature.delta(mel_spectrogram)
            mel_spectrogram_delta = torch.tensor(mel_spectrogram_delta, dtype=torch.float32)

            # Align all features to MFCC frames using interpolation
            mfcc_frames = mfcc.shape[1]
            features_to_align = [cqt, chroma, spectral_contrast, tonnetz, mel_spectrogram_delta]
            aligned_features = []
            for feature in features_to_align:
                feature_frames = feature.shape[1]
                if feature_frames != mfcc_frames:
                    feature = F.interpolate(feature.unsqueeze(0), size=mfcc_frames, mode="linear", align_corners=False).squeeze(0)
                aligned_features.append(feature)
            # Combine all features
            combined_features = torch.cat([mfcc] + aligned_features, dim=0)

            # Normalize features
            combined_features = self.normalize_features(combined_features)

            # Pad or truncate segments
            segments = []
            frames_per_segment = combined_features.shape[1] // num_segments
            for i in range(num_segments):
                start_frame = i * frames_per_segment
                end_frame = (i + 1) * frames_per_segment if i < num_segments - 1 else combined_features.shape[1]
                segment = combined_features[:, start_frame:end_frame]

                if segment.shape[1] < self.max_frames:
                    padding = self.max_frames - segment.shape[1]
                    segment = torch.nn.functional.pad(segment, (0, padding))
                elif segment.shape[1] > self.max_frames:
                    segment = segment[:, :self.max_frames]

                segments.append(segment)

            segments = torch.stack(segments)
            return segments, segment_labels

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, None

    def normalize_features(self, features):
        """
        Normalize features along the time axis (dim=1).
        """
        scaler = StandardScaler()
        features_np = features.numpy().T  # Transpose to shape (time_steps, num_features)
        normalized_features = scaler.fit_transform(features_np)
        return torch.tensor(normalized_features.T, dtype=torch.float32)


