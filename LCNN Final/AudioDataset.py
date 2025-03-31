class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_file, subset_ratio=0.02, max_frames=100, augment=True):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        self.labels_dict = np.load(label_file, allow_pickle=True).item()
        self.audio_files = [f for f in self.audio_files if os.path.splitext(f)[0] in self.labels_dict]
        num_files = int(len(self.audio_files) * subset_ratio)
        self.audio_files = self.audio_files[:num_files]
        self.audio_files = [os.path.join(audio_dir, f) for f in self.audio_files]

        # Track statistics
        self.original_stats = {'real': 0, 'fake': 0}
        self.augmented_stats = {'real': 0, 'fake': 0}
        
        # Count original distribution (real=1, spoof=0)
        total_segments = 0
        for audio_file in self.audio_files:
            file_id = os.path.splitext(os.path.basename(audio_file))[0]
            labels = self.labels_dict[file_id]
            # Convert string labels to integers
            labels = [int(l) for l in labels]
            total_segments += len(labels)
            self.original_stats['real'] += sum(1 for l in labels if l == 1)
            self.original_stats['fake'] += sum(1 for l in labels if l == 0)

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
        self.augment = augment

        print(f"Dataset initialized with {len(self.audio_files)} files")
        print(f"Original distribution - Real segments: {self.original_stats['real']}, Spoof segments: {self.original_stats['fake']}")

    def __len__(self):
        return len(self.audio_files)

    # Update the __getitem__ method in AudioDataset
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

            # Preprocess the audio input
            inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                wav2vec_embeddings = hubert_model(**inputs.to(device)).last_hidden_state.squeeze(0).cpu()

            # Track augmented distribution (real=1, spoof=0)
            if self.augment:
                # Update augmented stats for all segments in this file
                for label in segment_labels:
                    if label == 1:
                        self.augmented_stats['real'] += 1
                    else:
                        self.augmented_stats['fake'] += 1

                # Time stretching
                if random.random() < 0.5:
                    rate = random.uniform(0.8, 1.2)
                    waveform = torch.tensor(librosa.effects.time_stretch(waveform.numpy(), rate=rate))

                # Volume perturbation
                if random.random() < 0.5:
                    amp = random.uniform(0.5, 1.5)
                    waveform = waveform * amp

                # Add colored noise
                if random.random() < 0.3:
                    noise = np.random.normal(0, 0.005, len(waveform))
                    waveform = waveform + torch.tensor(noise, dtype=torch.float32)

                # Random frequency filtering
                if random.random() < 0.3:
                    cutoff = random.uniform(1000, 4000)
                    waveform = torch.tensor(librosa.effects.preemphasis(waveform.numpy(), coef=cutoff/16000))

            # Extract HuBERT Embeddings
            inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                hubert_outputs = hubert_model(**inputs)
                hubert_embeddings = hubert_outputs.last_hidden_state.squeeze(0).cpu()
                # Reduce HuBERT dimensions using max pooling and global average pooling
                hubert_embeddings = F.max_pool1d(hubert_embeddings.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
                hubert_embeddings = F.adaptive_avg_pool1d(hubert_embeddings.unsqueeze(0), 384).squeeze(0)

            # Compute MFCC
            mfcc = self.mfcc_transform(waveform)

            # Compute CQT using librosa
            cqt = librosa.cqt(waveform.numpy(), sr=16000, n_bins=84, bins_per_octave=12, hop_length=160)
            cqt = np.abs(cqt)  # Take the magnitude of the CQT
            cqt = torch.tensor(cqt, dtype=torch.float32)

            # Compute Chroma Features
            # Removing Chroma feature extraction

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
            features_to_align = [cqt, spectral_contrast, tonnetz, mel_spectrogram_delta, hubert_embeddings.T]  # Removed chroma
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
        scaler = StandardScaler()
        features_np = features.numpy()
        normalized_features = scaler.fit_transform(features_np)
        return torch.tensor(normalized_features, dtype=torch.float32)

    def get_statistics(self):
        return {
            'original': self.original_stats,
            'augmented': self.augmented_stats
        }
