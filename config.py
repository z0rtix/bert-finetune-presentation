import os
import numpy as np
import torch

RANDOM_SEED = 42
DEVICE = -1

FILLMASK_EN_MODEL = "bert-base-uncased"
FILLMASK_RU_MODEL = "bert-base-multilingual-cased"

TRAIN_SAMPLES = 1000
TEST_SAMPLES = 300
MAX_LEN = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8

RESULTS_DIR = "bert_demo_results"
CACHE_DIR = "bert_training_cache"
MODEL_DIR = "models/bert_imdb_finetuned"

BASELINE_ACC = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "attention_plots"), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)