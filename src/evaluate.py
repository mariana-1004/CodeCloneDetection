import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import EmbeddingDataset, CloneClassifier


def evaluate_classifier(
    test_embeddings,
    test_labels,
    device,
    model_path="clone_classifier.pt",
    batch_size=32,
    threshold=0.5,
    dropout=0.3
):
    test_dataset = EmbeddingDataset(test_embeddings, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = CloneClassifier(dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            raw_scores = model(embeddings)
            probabilities = torch.sigmoid(raw_scores)
            predictions = (probabilities >= threshold).int()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities
    }

    results_path = model_path.replace(".pt", "_results.pt")
    torch.save({"labels": all_labels, "predictions": all_predictions}, results_path)
    print(f"Evaluation results saved at: {results_path}")

    return results
