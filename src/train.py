import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import EmbeddingDataset, CloneClassifier

def compute_cosine_from_combined_embeddings(
        combined_embeddings,
        embedding_dim=768,
        chunk_size=100_000
        ):
    cosine_scores = []

    for start in range(0, len(combined_embeddings), chunk_size):
        end = min(start + chunk_size, len(combined_embeddings))

        batch = combined_embeddings[start:end].to("cpu").float()
        
        func1_embeddings = batch[:, :embedding_dim]
        func2_embeddings = batch[:, embedding_dim:embedding_dim*2]

        func1_embeddings = F.normalize(func1_embeddings, p=2, dim=1)
        func2_embeddings = F.normalize(func2_embeddings, p=2, dim=1)

        scores = (func1_embeddings * func2_embeddings).sum(dim=1)

        cosine_scores.append(scores.cpu())

    return torch.cat(cosine_scores)

def build_hard_negative_indices(
    train_embeddings,
    train_labels,
    embedding_dim=768,
    hard_negative_ratio=0.5,
    seed=42
):
    labels_cpu = train_labels.detach().cpu().long()

    positive_indices = torch.where(labels_cpu == 1)[0]
    negative_indices = torch.where(labels_cpu == 0)[0]

    number_of_hard_negatives = int(len(positive_indices) * hard_negative_ratio)
    number_of_hard_negatives = min(number_of_hard_negatives, len(negative_indices))

    if number_of_hard_negatives <= 0:
        print("No hard negatives selected.")
        return torch.arange(len(train_labels))

    cosine_scores = compute_cosine_from_combined_embeddings(
        train_embeddings,
        embedding_dim=embedding_dim
    )

    negative_scores = cosine_scores[negative_indices]

    _, top_positions = torch.topk(
        negative_scores,
        k=number_of_hard_negatives,
        largest=True
    )

    hard_negative_indices = negative_indices[top_positions]

    original_indices = torch.arange(len(train_labels))

    final_indices = torch.cat([
        original_indices,
        hard_negative_indices
    ])

    generator = torch.Generator()
    generator.manual_seed(seed)

    shuffled_positions = torch.randperm(len(final_indices), generator=generator)
    final_indices = final_indices[shuffled_positions]

    print("\n--- Hard Negative Mining ---")
    print(f"Original training examples: {len(original_indices)}")
    print(f"Positive examples:          {len(positive_indices)}")
    print(f"Negative examples:          {len(negative_indices)}")
    print(f"Added hard negatives:       {len(hard_negative_indices)}")
    print(f"Final training examples:    {len(final_indices)}")

    return final_indices

        

def train_classifier(
    train_embeddings,
    train_labels,
    device,
    batch_size=32,
    epochs=10,
    learning_rate=1e-4,
    dropout=0.3,
    model_save_path="clone_classifier.pt",
    use_hard_negative_mining=False,
    hard_negative_ratio=0.5
):
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)

    if use_hard_negative_mining:
        hard_negative_indices = build_hard_negative_indices(
            train_embeddings,
            train_labels,
            embedding_dim=768,
            hard_negative_ratio=hard_negative_ratio
        )

        train_dataset = Subset(
            train_dataset,
            hard_negative_indices.tolist()
        )

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
