import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def load_history(model_save_path):
    path = model_save_path.replace(".pt", "_history.pt")
    if not os.path.exists(path):
        print(f"History file not found: {path}")
        sys.exit(1)
    return torch.load(path, map_location="cpu")


def load_results(model_save_path):
    path = model_save_path.replace(".pt", "_results.pt")
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        return None
    return torch.load(path, map_location="cpu")


def plot_training_history(history, save_dir="."):
    epochs = range(1, len(history["acc"]) + 1)

    plt.figure()
    plt.ylim(bottom=0)
    plt.plot(epochs, history["acc"], "o-", color="blue", label="Train accuracy")
    plt.title("Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    path = os.path.join(save_dir, "accuracy_curves.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.show()

    plt.figure()
    plt.ylim(bottom=0)
    plt.plot(epochs, history["loss"], "o-", color="blue", label="Train loss")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(labels, predictions, save_dir="."):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions,
        cmap="Blues",
        colorbar=False,
        ax=ax,
        display_labels=["No clone", "Clone"]
    )
    ax.set_title("Confusion Matrix")
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.show()


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import MODEL_SAVE_PATH

    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not save_dir:
        save_dir = "."

    history = load_history(MODEL_SAVE_PATH)
    plot_training_history(history, save_dir=save_dir)

    results = load_results(MODEL_SAVE_PATH)
    if results is not None:
        plot_confusion_matrix(results["labels"], results["predictions"], save_dir=save_dir)


if __name__ == "__main__":
    main()
