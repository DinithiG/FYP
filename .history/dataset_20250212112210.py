class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_file, subset_ratio=0.7, max_frames=100):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.labels_dict = np.load(label_file, allow_pickle=True).item()
        self.audio_files = [f for f in self.audio_files if os.path.splitext(f)[0] in self.labels_dict]
        num_files = int(len(self.audio_files) * subset_ratio)
        self.audio_files = self.audio_files[:num_files]
        self.audio_files = [os.path.join(audio_dir, f) for f in self.audio_files]
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 80,
                "center": True,
                "normalized": True
            }
        )
      self.max_frames = max_frames  # Define a maximum number of frames
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
            mfcc = self.mfcc_transform(waveform)

            # Pad or truncate MFCC segments
            segments = []
            frames_per_segment = mfcc.shape[1] // num_segments
            for i in range(num_segments):
                start_frame = i * frames_per_segment
                 end_frame = (i + 1) * frames_per_segment if i < num_segments - 1 else mfcc.shape[1]
                segment = mfcc[:, start_frame:end_frame]

                # Pad or truncate the segment
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
