import warnings
warnings.filterwarnings("ignore")

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging as hf_logging
from datasets import load_dataset

from config import (TRAIN_SAMPLES, TEST_SAMPLES, MAX_LEN, NUM_EPOCHS, BATCH_SIZE,
                    CACHE_DIR, MODEL_DIR, DEVICE, RANDOM_SEED)
from utils import prepare_imdb_data, tokenize_dataset, compute_metrics

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_and_save():
    print("=== ОБУЧЕНИЕ BERT НА IMDb ===")
    train_raw, test_raw = prepare_imdb_data(TRAIN_SAMPLES, TEST_SAMPLES)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_tok = tokenize_dataset(train_raw, tokenizer, MAX_LEN)
    test_tok = tokenize_dataset(test_raw, tokenizer, MAX_LEN)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(CACHE_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(CACHE_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=False,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics
    )
    print(">>> Начинаем обучение (CPU, ~5-10 минут)...")
    trainer.train()
    
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"[OK] Модель сохранена в {MODEL_DIR}")
    return trainer, test_tok

if __name__ == "__main__":
    train_and_save()