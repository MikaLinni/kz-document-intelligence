from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


CORPUS_PATH = DATA_DIR / "legal_chunks_20k.parquet"

TRAIN_PATH = DATA_DIR / "retrieval_train.parquet"
DEV_PATH = DATA_DIR / "retrieval_dev.parquet"
TEST_PATH = DATA_DIR / "retrieval_test.parquet"


TRAIN_RECORDS_PATH = DATA_DIR / "bge_m3_training_records.jsonl"

BEST_MODEL_DIR = MODELS_DIR / "bge_m3_legal_lora_best"

TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"

DEV_METRICS_PATH = RESULTS_DIR / "dev_metrics.json"

TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"


MODEL_NAME = "BAAI/bge-m3"


SEED = 42


NEGATIVES_PER_QUERY = 3

BM25_POOL_SIZE = 40

BM25_SKIP_TOP = 5


LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


EPOCHS = 5

EARLY_STOPPING_PATIENCE = 2


LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

TEMPERATURE = 0.05


GRADIENT_ACCUMULATION_STEPS = 8


QUERY_MAX_LENGTH = 128

PASSAGE_MAX_LENGTH = 320


DEV_CORPUS_BATCH_SIZE = 16

DEV_QUERY_BATCH_SIZE = 32