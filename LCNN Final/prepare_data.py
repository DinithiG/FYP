def prepare_data(dataset):
    all_segments = []
    all_labels = []
    
    # Reset augmented stats before counting
    dataset.augmented_stats = {'real': 0, 'fake': 0}

    for i in range(len(dataset)):
        segments, labels = dataset[i]
        if segments is not None and labels is not None:
            # Count augmented statistics for each segment
            if dataset.augment:
                for label in labels:
                    if label == 1:
                        dataset.augmented_stats['real'] += 1
                    else:
                        dataset.augmented_stats['fake'] += 1
            
            all_segments.append(segments)
            all_labels.append(labels)

    if not all_segments:
        raise ValueError("No features were extracted. Check the audio files and feature extraction process.")

    segments = torch.cat(all_segments, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"\nFinal shapes:")
    print(f"All segments: {segments.shape}")
    print(f"All labels: {labels.shape}")

    # Print augmented statistics after processing all files
    if dataset.augment:
        print("\nAugmented Statistics:")
        print(f"Real segments: {dataset.augmented_stats['real']}")
        print(f"Spoof segments: {dataset.augmented_stats['fake']}")

    return segments, labels

