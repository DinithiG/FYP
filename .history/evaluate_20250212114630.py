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

def train_and_evaluate(train_segments, train_labels, dev_segments, dev_labels, num_epochs=50, batch_size=32, learning_rate=0.001):
    train_dataset = torch.utils.data.TensorDataset(train_segments, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = torch.utils.data.TensorDataset(dev_segments, dev_labels)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model with updated input_dim
    model = LCNN(input_dim=124 + 12 + 7 + 6 + 128).to(device)  # Include all features
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5, verbose=True)

    best_f1 = 0.0
    best_model_state_dict = None
    early_stopping_patience = 10
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_segments, batch_labels in train_loader:
            batch_segments = batch_segments.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_segments)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        dev_loss, accuracy, f1, report, cm = evaluate(model, dev_loader, criterion)
        scheduler.step(dev_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {accuracy:.4f}, Dev F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model_state_dict = model.state_dict()
            print("Best model updated!")
            early_stopping_counter = 0  # Reset the counter
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
    print("\nTraining Complete. Best Dev F1:", best_f1)
    torch.save(best_model_state_dict, "/kaggle/working/trained_model_best.pth")
    print("Best model saved to /kaggle/working/trained_model_best.pth")
    return best_model_state_dict

