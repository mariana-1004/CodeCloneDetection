import os
import torch

from config import (
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DROPOUT,
    TRAIN_SAMPLE_SIZE,
    TEST_SAMPLE_SIZE,
    THRESHOLD,
    TRAIN_EMBEDDINGS_PATH,
    TRAIN_LABELS_PATH,
    TEST_EMBEDDINGS_PATH,
    TEST_LABELS_PATH,
    MODEL_SAVE_PATH,
)

from data import load_bigclonebench, preprocess, sample_dataframe
from embeddings import (
    load_codebert,
    generate_embeddings,
    save_embeddings,
    load_embeddings
)
from train import train_classifier
from evaluate import evaluate_classifier


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def embeddings_exist():
    required_files = [
        TRAIN_EMBEDDINGS_PATH,
        TRAIN_LABELS_PATH,
        TEST_EMBEDDINGS_PATH,
        TEST_LABELS_PATH
    ]

    return all(os.path.exists(file) for file in required_files)


def main():
    device = get_device()
    print("Using device:", device)

    if not embeddings_exist():
        print("Embeddings not found. Generating embeddings...")

        train_df, test_df = load_bigclonebench("google/code_x_glue_cc_clone_detection_big_clone_bench")

        train_df = sample_dataframe(train_df, TRAIN_SAMPLE_SIZE)
        test_df = sample_dataframe(test_df, TEST_SAMPLE_SIZE)

        train_func1, train_func2, train_labels = preprocess(train_df)
        test_func1, test_func2, test_labels = preprocess(test_df)

        tokenizer, codebert = load_codebert("microsoft/codebert-base", device)

        train_embeddings, train_labels_tensor = generate_embeddings(
            train_func1,
            train_func2,
            train_labels,
            tokenizer,
            codebert,
            device,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE
        )

        test_embeddings, test_labels_tensor = generate_embeddings(
            test_func1,
            test_func2,
            test_labels,
            tokenizer,
            codebert,
            device,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE
        )

        save_embeddings(
            train_embeddings,
            train_labels_tensor,
            TRAIN_EMBEDDINGS_PATH,
            TRAIN_LABELS_PATH
        )

        save_embeddings(
            test_embeddings,
            test_labels_tensor,
            TEST_EMBEDDINGS_PATH,
            TEST_LABELS_PATH
        )

        print("Embeddings generated and saved successfully.")

    else:
        print("Embeddings found. Loading existing embeddings...")

    train_embeddings, train_labels = load_embeddings(
         TRAIN_EMBEDDINGS_PATH,
         TRAIN_LABELS_PATH
    )
    
    test_embeddings, test_labels = load_embeddings(
         TEST_EMBEDDINGS_PATH,
         TEST_LABELS_PATH
    )

    train_classifier(
         train_embeddings,
         train_labels,
         device,
         batch_size=BATCH_SIZE,
         epochs=EPOCHS,
         learning_rate=LEARNING_RATE,
         dropout=DROPOUT,
         model_save_path=MODEL_SAVE_PATH
    )
    
    evaluate_classifier(
         test_embeddings,
         test_labels,
         device,
         model_path=MODEL_SAVE_PATH,
         batch_size=BATCH_SIZE,
         threshold=THRESHOLD,
         dropout=DROPOUT
    )


if __name__ == "__main__":
    main()
