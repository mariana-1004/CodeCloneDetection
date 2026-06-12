import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# EXPERIMENTS
# =====================================

EXPERIMENTS = {
    "Baseline": {
        "history": "../models/baseline/clone_classifier_baseline_history.pt",
        "results": "../models/baseline/clone_classifier_baseline_results.pt"
    },
    "BigDNN": {
        "history": "../models/E1_bigDNN/clone_classifier_bigDNN_history.pt",
        "results": "../models/E1_bigDNN/clone_classifier_bigDNN_results.pt"
    },
    "BigDNN2": {
        "history": "../models/E2_bigDNN2/clone_classifier_bigDNN2_history.pt",
        "results": "../models/E2_bigDNN2/clone_classifier_bigDNN2_results.pt"
    },
    "HardNegative": {
        "history": "../models/E3_HardNegatives/clone_classifier_hardNegative_history.pt",
        "results": "../models/E3_HardNegatives/clone_classifier_hardNegative_results.pt"
    }
}


# =====================================
# LOADERS
# =====================================

def load_history(path):

    if not os.path.exists(path):
        print(f"History file not found: {path}")
        return None

    return torch.load(path, map_location="cpu")


def load_results(path):

    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        return None

    return torch.load(path, map_location="cpu")


# =====================================
# ACCURACY CURVES
# =====================================

def plot_accuracy_comparison(experiments, save_dir="."):

    plt.figure(figsize=(10, 6))

    for name, files in experiments.items():

        history = load_history(files["history"])

        if history is None:
            continue

        epochs = range(1, len(history["acc"]) + 1)

        plt.plot(
            epochs,
            history["acc"],
            marker="o",
            linewidth=2,
            label=name
        )

    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(
        save_dir,
        "accuracy_comparison.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


# =====================================
# LOSS CURVES
# =====================================

def plot_loss_comparison(experiments, save_dir="."):

    plt.figure(figsize=(10, 6))

    for name, files in experiments.items():

        history = load_history(files["history"])

        if history is None:
            continue

        epochs = range(1, len(history["loss"]) + 1)

        plt.plot(
            epochs,
            history["loss"],
            marker="o",
            linewidth=2,
            label=name
        )

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = os.path.join(
        save_dir,
        "loss_comparison.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


# =====================================
# METRICS TABLE
# =====================================

def compute_metrics(experiments):

    rows = []

    for name, files in experiments.items():

        results = load_results(files["results"])

        if results is None:
            continue

        labels = results["labels"]
        predictions = results["predictions"]

        accuracy = accuracy_score(labels, predictions)

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        rows.append([
            name,
            accuracy,
            precision,
            recall,
            f1
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    )

    return df


def print_metrics_table(df):

    print("\n")
    print("=" * 80)
    print(df.round(4).to_string(index=False))
    print("=" * 80)


# =====================================
# BAR CHART
# =====================================

def plot_metrics_comparison(df, save_dir="."):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    models = df["Model"].tolist()

    x = np.arange(len(metrics))
    width = 0.18

    plt.figure(figsize=(12, 6))

    for i, (_, row) in enumerate(df.iterrows()):

        values = [
            row["Accuracy"],
            row["Precision"],
            row["Recall"],
            row["F1"]
        ]

        plt.bar(
            x + i * width,
            values,
            width,
            label=row["Model"]
        )

    plt.xticks(
        x + width * (len(models) - 1) / 2,
        metrics
    )

    plt.ylabel("Score")
    plt.ylim(0, 1.0)

    plt.title(
        "Final Metrics Comparison"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3
    )

    output_path = os.path.join(
        save_dir,
        "metrics_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()


# =====================================
# MAIN
# =====================================

def main():

    output_dir = "../models"

    # Curvas de entrenamiento
    plot_accuracy_comparison(
        EXPERIMENTS,
        save_dir=output_dir
    )

    plot_loss_comparison(
        EXPERIMENTS,
        save_dir=output_dir
    )

    # Métricas finales
    df = compute_metrics(EXPERIMENTS)

    print_metrics_table(df)

    csv_path = os.path.join(
        output_dir,
        "metrics_comparison.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(f"Saved: {csv_path}")

    # Barras comparativas
    plot_metrics_comparison(
        df,
        save_dir=output_dir
    )


if __name__ == "__main__":
    main()