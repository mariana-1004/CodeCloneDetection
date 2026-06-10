import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx]
        }


class CloneClassifier(nn.Module):
    def __init__(self, input_dim=768 * 4, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, combined_embeddings):
        raw_scores = self.classifier(combined_embeddings)
        return raw_scores.squeeze(1)
