def prepare_data(dataset):
    all_segments = []
    all_labels = []

    for i in range(len(dataset)):
        segments, labels = dataset[i]
        if segments is not None and labels is not None:
            print(f"File {i} - Segments shape: {segments.shape}, Labels shape: {labels.shape}")
            all_segments.append(segments)
            all_labels.append(labels)

    if not all_segments:
        raise ValueError("No features were extracted. Check the audio files and feature extraction process.")

    segments = torch.cat(all_segments, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"\nFinal shapes:")
    print(f"All segments: {segments.shape}")
    print(f"All labels: {labels.shape}")

    return segments, labels