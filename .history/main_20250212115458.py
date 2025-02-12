# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # --- Load Train, Dev, and Eval Datasets (with subset) ---
    train_audio_dir = "/kaggle/input/partialspoof/NewData/train2/con_wav"
    train_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/train_seglab_0.16.npy"
    dev_audio_dir = "/kaggle/input/partialspoof/NewData/dev/con_wav"
    dev_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/dev_seglab_0.16.npy"
    eval_audio_dir = "/kaggle/input/partialspoof/NewData/eval/con_wav"
    eval_label_file = "/kaggle/input/partialspoof/NewData/segment_labels/eval_seglab_0.16.npy"

    subset_ratio = 0.02
    print("Initializing datasets (with subset ratio = 0.02)...")

    train_dataset = AudioDataset(train_audio_dir, train_label_file, subset_ratio=1.0, max_frames=100)
    dev_dataset = AudioDataset(dev_audio_dir, dev_label_file, subset_ratio=0.5, max_frames=100)
    eval_dataset = AudioDataset(eval_audio_dir, eval_label_file, subset_ratio=0.05, max_frames=100)

    print("\nPreparing data...")
    train_segments, train_labels = prepare_data(train_dataset)
    dev_segments, dev_labels = prepare_data(dev_dataset)
    eval_segments, eval_labels = prepare_data(eval_dataset)
    print("\nStarting training...")
    best_model_weights = train_and_evaluate(
        train_segments, train_labels, dev_segments, dev_labels, num_epochs=20, batch_size=64, learning_rate=0.0005
    )

    # --- Final Evaluation on Eval Set ---
    model = LCNN(input_dim=124 + 12 + 7 + 6 + 128).to(device)  # Include all features
    model.load_state_dict(best_model_weights)
    model.eval()

    eval_dataset_final = torch.utils.data.TensorDataset(eval_segments, eval_labels)
    eval_loader = DataLoader(eval_dataset_final, batch_size=64, shuffle=False)

    eval_loss, accuracy, f1, report, cm = evaluate(model, eval_loader, nn.BCELoss())
    print("\n--- Final Evaluation on Held-out Eval Set ---")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Accuracy: {accuracy:.4f}")
    print(f"Eval F1 Score: {f1:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # Advanced Metrics
    def calculate_eer(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer

    def get_scores(model, data_loader):
        model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_segments, batch_labels in data_loader:
                batch_segments = batch_segments.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_segments)
                scores = outputs.squeeze().cpu().numpy()

                all_scores.extend(scores)
                all_labels.extend(batch_labels.cpu().numpy())

        return np.array(all_labels), np.array(all_scores)

    labels, scores = get_scores(model, eval_loader)
    eer = calculate_eer(labels, scores)
    print(f"EER: {eer:.4f}")

    def calculate_min_tDCF(labels, scores, p_target=0.05, c_miss=1, c_fa=1):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        tDCF = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
        min_tDCF = np.min(tDCF)
        return min_tDCF

    min_tDCF = calculate_min_tDCF(labels, scores)
    print(f"min-TCDF: {min_tDCF:.4f}")

    def find_optimal_threshold(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
        return eer_threshold

    optimal_threshold = find_optimal_threshold(labels, scores)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

