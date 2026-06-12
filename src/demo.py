import os
import torch

from config import (
    MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, DROPOUT, THRESHOLD,
    TRAIN_EMBEDDINGS_PATH, TRAIN_LABELS_PATH, MODEL_SAVE_PATH,
)
from embeddings import load_codebert, generate_embeddings, load_embeddings
from model import CloneClassifier
from train import train_classifier


DEMO_PAIRS = [
    {
        "func1": (
            "public int add(int a, int b) {\n"
            "    return a + b;\n"
            "}"
        ),
        "func2": (
            "public int add(int a, int b) {\n"
            "    return a + b;\n"
            "}"
        ),
        "label": 1,
        "description": "Exact duplicate",
    },
    {
        "func1": (
            "public int add(int a, int b) {\n"
            "    return a + b;\n"
            "}"
        ),
        "func2": (
            "public int sum(int x, int y) {\n"
            "    return x + y;\n"
            "}"
        ),
        "label": 1,
        "description": "Semantic clone (renamed)",
    },
    {
        "func1": (
            "public int max(int a, int b) {\n"
            "    return a > b ? a : b;\n"
            "}"
        ),
        "func2": (
            "public int maxValue(int x, int y) {\n"
            "    if (x > y) {\n"
            "        return x;\n"
            "    } else {\n"
            "        return y;\n"
            "    }\n"
            "}"
        ),
        "label": 1,
        "description": "Semantic clone (ternary vs if-else)",
    },
    {
        "func1": (
            "public long calcular(int n) { return java.util.stream.LongStream.rangeClosed(1, n).reduce(1, (a, b) -> a * b); }"
        ),
        "func2": (
            "public long procesar(int n) { long[] b = new long[n + 1]; b[0] = 1; int i = 1; while (i <= n) { b[i] = b[i - 1] * i++; } return b[n]; }"
        ),
        "label": 1,
        "description": "Harder semantic clone (factorial with streams vs iterative)",
    },
    {
        "func1": (
            "public boolean isPalindrome(String s) {\n"
            "    String rev = new StringBuilder(s).reverse().toString();\n"
            "    return s.equals(rev);\n"
            "}"
        ),
        "func2": (
            "public double circleArea(double radius) {\n"
            "    return Math.PI * radius * radius;\n"
            "}"
        ),
        "label": 0,
        "description": "Non-clone (palindrome vs area)",
    },
    {
        "func1": (
            "public void sortArray(int[] arr) {\n"
            "    java.util.Arrays.sort(arr);\n"
            "}"
        ),
        "func2": (
            "public int fibonacci(int n) {\n"
            "    if (n <= 1) return n;\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2);\n"
            "}"
        ),
        "label": 0,
        "description": "Non-clone (sort vs fibonacci)",
    },
]


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def one_line(code, max_chars=65):
    flat = code.replace("\n", " ").replace("    ", " ").strip()
    if len(flat) > max_chars:
        return flat[: max_chars - 3] + "..."
    return flat


def main():
    device = get_device()
    print("Using device:", device)

    banner = " Code Clone Detection Demo "
    print(f"\n{'=' * 60}")
    print(f"{banner:^60}")
    print(f"{'=' * 60}\n")

    if not os.path.exists(MODEL_SAVE_PATH):
       print("No trained model found. Please run 'train.py' first to train the classifier.") 

    print("\nLoading CodeBERT...")
    tokenizer, codebert = load_codebert("microsoft/codebert-base", device)

    print("Loading trained classifier...")
    model = CloneClassifier(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    func1_list = [p["func1"] for p in DEMO_PAIRS]
    func2_list = [p["func2"] for p in DEMO_PAIRS]
    labels = [p["label"] for p in DEMO_PAIRS]

    print("Generating embeddings for demo pairs...")
    embeddings, _ = generate_embeddings(
        func1_list, func2_list, labels,
        tokenizer, codebert, device,
        max_length=MAX_LENGTH,
        batch_size=len(DEMO_PAIRS),
    )

    print("Running inference...")
    with torch.no_grad():
        raw_scores = model(embeddings.to(device))
        probabilities = torch.sigmoid(raw_scores)
        predictions = (probabilities >= THRESHOLD).int()

    correct = 0
    total = len(DEMO_PAIRS)

    print(f"\n{'─' * 72}")
    for i, pair in enumerate(DEMO_PAIRS):
        truth = pair["label"]
        pred = predictions[i].item()
        conf = probabilities[i].item()
        is_correct = pred == truth
        if is_correct:
            correct += 1

        truth_label = "CLONE" if truth == 1 else "NON-CLONE"
        pred_label = "CLONE" if pred == 1 else "NON-CLONE"
        mark = "CORRECT" if is_correct else "WRONG"

        print(f"\n  Pair {i + 1}: {pair['description']}")
        print(f"  F1:  {one_line(pair['func1'])}")
        print(f"  F2:  {one_line(pair['func2'])}")
        print(f"  Truth: {truth_label:>10}  |  Pred: {pred_label:>10}  "
              f"|  Confidence: {conf:.4f}  |  [{mark}]")

    print(f"\n{'─' * 72}")
    print(f"\n  SUMMARY: {correct}/{total} correct  "
          f"({100.0 * correct / total:.1f}% accuracy)\n")


if __name__ == "__main__":
    main()
