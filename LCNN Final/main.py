
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model directly
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert_model.eval()  # Set to evaluation mode


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_audio_dir = "/kaggle/input/high-pass-filtered/filtered_data/train2_filtered/con_wav"
    train_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/train_seglab_0.16.npy"
    dev_audio_dir = "/kaggle/input/high-pass-filtered/filtered_data/dev_filtered/con_wav"
    dev_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/dev_seglab_0.16.npy"
    eval_audio_dir = "/kaggle/input/high-pass-filtered/filtered_data/eval_filtered/con_wav"
    eval_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/eval_seglab_0.16.npy"

    print("Initializing datasets...")
    train_dataset = AudioDataset(train_audio_dir, train_label_file, subset_ratio=0.1, max_frames=100, augment=True)
    dev_dataset = AudioDataset(dev_audio_dir, dev_label_file, subset_ratio=0.01, max_frames=100, augment=False)
    eval_dataset = AudioDataset(eval_audio_dir, eval_label_file, subset_ratio=0.01, max_frames=100, augment=False)

    train_stats = train_dataset.get_statistics()
    print("\nDataset Statistics:")
    print("Original Distribution:")
    print(f"Real segments: {train_stats['original']['real']}")
    print(f"Spoof segments: {train_stats['original']['fake']}")

    print("\nPreparing data...")
    train_segments, train_labels = prepare_data(train_dataset)
    dev_segments, dev_labels = prepare_data(dev_dataset)
    eval_segments, eval_labels = prepare_data(eval_dataset)

    print("\nStarting training...")
    best_model_weights = train_and_evaluate(
        train_segments, train_labels, dev_segments, dev_labels, num_epochs=50, batch_size=32, learning_rate=0.0008, device=device
    )

    model = LCNN(input_dim=124 + 7 + 6 + 128 + 384).to(device)  # Update this to match the new dimensions
    model.load_state_dict(best_model_weights)
    model.eval()

    eval_dataset_final = torch.utils.data.TensorDataset(eval_segments, eval_labels)
    eval_loader = DataLoader(eval_dataset_final, batch_size=64, shuffle=False)

    eval_loss, accuracy, f1, report, cm = evaluate(model, eval_loader, FocalLoss(alpha=0.35, gamma=2.5, label_smoothing=0.1), device)
    print("\n--- Final Evaluation on Held-out Eval Set ---")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Accuracy: {accuracy:.4f}")
    print(f"Eval F1 Score: {f1:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    labels, scores = get_scores(model, eval_loader, device)
    eer = calculate_eer(labels, scores)
    print(f"EER: {eer:.4f}")

    min_tDCF = calculate_min_tDCF(labels, scores)
    print(f"min-TCDF: {min_tDCF:.4f}")

    optimal_threshold = find_optimal_threshold(labels, scores)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    plt.figure(figsize=(15, 10))
    visualize_features(eval_segments, eval_labels)
    plt.savefig('/kaggle/working/feature_visualization.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Spoof'],
                yticklabels=['Genuine', 'Spoof'])
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/kaggle/working/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    plot_roc_curve(labels, scores, '/kaggle/working/roc_curve.png')
    plot_eer_curve(labels, scores, '/kaggle/working/eer_curve.png')

    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, eval_segments, eval_labels, device)

    print("\nAll visualization plots have been saved!")