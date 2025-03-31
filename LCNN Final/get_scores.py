def get_scores(model, data_loader, device):
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