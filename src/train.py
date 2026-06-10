import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            raw_scores = model(embeddings)
            loss = criterion(raw_scores, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    return model
