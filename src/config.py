MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT = 0.3

TRAIN_SAMPLE_SIZE = None
TEST_SAMPLE_SIZE = None

THRESHOLD = 0.5

TRAIN_EMBEDDINGS_PATH = "../data/embeddings/train_embeddings_full.pt"
TRAIN_LABELS_PATH = "../data/embeddings/train_labels_full.pt"
TEST_EMBEDDINGS_PATH = "../data/embeddings/test_embeddings_full.pt"
TEST_LABELS_PATH = "../data/embeddings/test_labels_full.pt"

MODEL_SAVE_PATH = "../models/clone_classifier_full.pt"
