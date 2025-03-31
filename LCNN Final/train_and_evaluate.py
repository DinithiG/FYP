def train_and_evaluate(train_segments, train_labels, dev_segments, dev_labels, num_epochs=50, batch_size=32, learning_rate=0.001, device='cpu'):
    train_dataset = torch.utils.data.TensorDataset(train_segments, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = torch.utils.data.TensorDataset(dev_segments, dev_labels)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = LCNN(input_dim=124 + 7 + 6 + 128 + 384).to(device)  # Removed 12 for Chroma
    # Remove the recursive call and loading of weights
    # model.load_state_dict(best_model_weights)
    
    criterion = FocalLoss(alpha=0.35, gamma=2.5, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    best_f1 = 0.0
    best_model_state_dict = None
    early_stopping_patience = 10
    early_stopping_counter = 0

    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_segments, batch_labels in train_loader:
            batch_segments = batch_segments.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_segments)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            predictions = (outputs.squeeze() >= 0.5).float()
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_accuracy = train_correct / train_total
        avg_train_loss = total_train_loss / len(train_loader)
        dev_loss, dev_accuracy, f1, report, cm = evaluate(model, dev_loader, criterion, device)
        scheduler.step()

        train_losses.append(avg_train_loss)
        dev_losses.append(dev_loss)
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}, Dev F1: {f1:.4f}")
        print("Dev Confusion Matrix:\n", cm, "\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model_state_dict = model.state_dict()
            print("Best model updated!")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    plot_training_curves(train_losses, dev_losses, train_accuracies, dev_accuracies, '/kaggle/working/training_curves.png')

    print("\nTraining Complete. Best Dev F1:", best_f1)
    torch.save(best_model_state_dict, "/kaggle/working/trained_model_best.pth")
    print("Best model saved to /kaggle/working/trained_model_best.pth")
    return best_model_state_dict


def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def calculate_min_tDCF(labels, scores, p_target=0.05, c_miss=1, c_fa=1):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    tDCF = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    min_tDCF = np.min(tDCF)
    return min_tDCF


def find_optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
    return eer_threshold


def visualize_features(model_input, labels, num_samples=5):
    if torch.is_tensor(model_input):
        model_input = model_input.cpu()
    if torch.is_tensor(labels):
        labels = labels.cpu()
    
    indices = np.random.choice(len(model_input), num_samples, replace=False)
    
    feature_names = [
        'MFCC (40)',              
        'CQT (84)',              
        # Removed 'Chroma (12)',           
        'Spectral Contrast (7)', 
        'Tonnetz (6)',           
        'Mel Spectrogram Delta (128)', 
        'Wav2Vec (384)'          
    ]
    
    feature_boundaries = [40, 124, 131, 137, 265, 649]  # Updated boundaries
    
    plt.figure(figsize=(15, 10))
    
    for idx, sample_idx in enumerate(indices):
        sample = model_input[sample_idx].numpy()
        label = labels[sample_idx].item()
        
        plt.subplot(num_samples, 1, idx + 1)
        
        start_idx = 0
        for feat_idx, (name, end_idx) in enumerate(zip(feature_names, feature_boundaries)):
            feature_section = sample[start_idx:end_idx]
            
            if len(feature_section.shape) > 1:
                feature_section = np.mean(feature_section, axis=1)
            
            if len(feature_section) > 0:
                feature_section = (feature_section - feature_section.min()) / (feature_section.max() - feature_section.min() + 1e-6)
            
            plt.plot(range(start_idx, end_idx), feature_section, 
                    label=name, alpha=0.7)
            start_idx = end_idx
        
        if idx == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'Sample {idx + 1} (Label: {"Genuine" if label == 1 else "Spoof"})')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_training_curves(train_losses, dev_losses, train_accuracies, dev_accuracies, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(dev_losses, label='Dev Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(dev_accuracies, label='Dev Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(labels, scores, save_path):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Add EER point
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = thresholds[np.nanargmin(np.abs(tpr - fpr))]
    plt.plot(eer, eer, 'ro', markersize=8, label=f'EER = {eer:.3f}')
    
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_eer_curve(labels, scores, save_path):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fpr, label='FPR')
    plt.plot(thresholds, fnr, label='FNR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Equal Error Rate (EER) Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def analyze_feature_importance(model, eval_segments, eval_labels, device):
    feature_groups = {
        'MFCC': (0, 40),
        'CQT': (40, 124),
        # Removed 'Chroma': (124, 136),
        'Spectral Contrast': (124, 131),  # Updated indices
        'Tonnetz': (131, 137),  # Updated indices
        'Mel Spectrogram Delta': (137, 265),  # Updated indices
        'HuBERT': (265, 649)  # Updated indices
    }
    
    eval_dataset = torch.utils.data.TensorDataset(eval_segments, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    baseline_labels, baseline_scores = get_scores(model, eval_loader, device)
    baseline_eer = calculate_eer(baseline_labels, baseline_scores)
    baseline_min_tdcf = calculate_min_tDCF(baseline_labels, baseline_scores)
    
    print("\nFeature Importance Analysis:")
    print(f"Baseline EER: {baseline_eer:.4f}")
    print(f"Baseline min-tDCF: {baseline_min_tdcf:.4f}")
    
    feature_importance = {}
    for name, (start_idx, end_idx) in feature_groups.items():
        modified_segments = eval_segments.clone()
        modified_segments[:, start_idx:end_idx] = 0
        
        modified_dataset = torch.utils.data.TensorDataset(modified_segments, eval_labels)
        modified_loader = DataLoader(modified_dataset, batch_size=64, shuffle=False)
        
        modified_labels, modified_scores = get_scores(model, modified_loader, device)
        modified_eer = calculate_eer(modified_labels, modified_scores)
        modified_min_tdcf = calculate_min_tDCF(modified_labels, modified_scores)
        
        eer_degradation = modified_eer - baseline_eer
        tdcf_degradation = modified_min_tdcf - baseline_min_tdcf
        
        feature_importance[name] = {
            'EER degradation': eer_degradation,
            'min-tDCF degradation': tdcf_degradation
        }
        
        print(f"\n{name}:")
        print(f"EER degradation: {eer_degradation:.4f}")
        print(f"min-tDCF degradation: {tdcf_degradation:.4f}")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    eer_degradations = [v['EER degradation'] for v in feature_importance.values()]
    plt.bar(feature_groups.keys(), eer_degradations)
    plt.title('Feature Importance (EER Degradation)')
    plt.xticks(rotation=45)
    plt.ylabel('EER Degradation')
    
    plt.subplot(1, 2, 2)
    tdcf_degradations = [v['min-tDCF degradation'] for v in feature_importance.values()]
    plt.bar(feature_groups.keys(), tdcf_degradations)
    plt.title('Feature Importance (min-tDCF Degradation)')
    plt.xticks(rotation=45)
    plt.ylabel('min-tDCF Degradation')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return feature_importance
