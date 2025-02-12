def evaluate(model, data_loader, criterion):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch_segments, batch_labels in data_loader:
            batch_segments = batch_segments.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_segments)
            loss = criterion(outputs.squeeze(), batch_labels)

            total_loss += loss.item()
            predictions = (outputs.squeeze() >= 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    report = classification_report(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    return avg_loss, accuracy, f1, report, cm

