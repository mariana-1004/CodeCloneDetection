import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EmbeddingDataset, CloneClassifier


def train_classifier(
    train_embeddings,
    train_labels,
    device,
    batch_size=32,
    epochs=10,
    learning_rate=1e-4,
    dropout=0.3,
    model_save_path="clone_classifier.pt"
):
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = CloneClassifier(dropout=dropout).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            raw_scores = model(embeddings)
            loss = criterion(raw_scores, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(raw_scores) >= 0.5).int()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total

        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}"
        )

    torch.save(model.state_dict(), model_save_path)
    torch.save(history, model_save_path.replace(".pt", "_history.pt"))
    print(f"Model saved at: {model_save_path}")
    print(f"Training history saved at: {model_save_path.replace('.pt', '_history.pt')}")

    return model, history
