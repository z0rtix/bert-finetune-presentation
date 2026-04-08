import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset

from config import RESULTS_DIR, MODEL_DIR, BASELINE_ACC, TEST_SAMPLES, RANDOM_SEED
from utils import (print_header, run_fill_mask_en, run_fill_mask_ru, prepare_imdb_data,
                   tokenize_dataset, plot_accuracy_comparison, plot_training_curves,
                   plot_confusion_matrix, plot_classification_report, plot_prediction_confidence,
                   visualize_attention, show_prediction_examples)

def evaluate():
    print_header("ЗАГРУЗКА СОХРАНЁННОЙ МОДЕЛИ")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    run_fill_mask_en()
    run_fill_mask_ru()
    
    _, test_raw = prepare_imdb_data(10, TEST_SAMPLES)
    tokenizer_orig = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_tok = tokenize_dataset(test_raw, tokenizer_orig, max_len=128)
    
    from transformers import Trainer
    import torch
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_tok, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    true_labels = np.array(all_labels)
    pred_labels = np.array(all_preds)
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='binary')
    test_prec = precision_score(true_labels, pred_labels, average='binary')
    test_rec = recall_score(true_labels, pred_labels, average='binary')
    
    print(f"\n[OK] Результаты fine-tuning (загруженная модель):")
    print(f"   Accuracy : {test_acc:.2%}")
    print(f"   F1-score : {test_f1:.3f}")
    print(f"   Precision: {test_prec:.3f}")
    print(f"   Recall   : {test_rec:.3f}")
    
    print_header("ГЕНЕРАЦИЯ ГРАФИКОВ")
    plot_accuracy_comparison(BASELINE_ACC, test_acc,
                             save_path=os.path.join(RESULTS_DIR, "accuracy_comparison.png"))
    
    print("[INFO] Кривые обучения не генерируются при загрузке модели (нужны логи обучения).")
    print("      Для их получения запустите train.py с флагом логирования или используйте оригинальный скрипт.")
    
    plot_confusion_matrix(true_labels, pred_labels,
                          save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_classification_report(true_labels, pred_labels,
                               save_path=os.path.join(RESULTS_DIR, "classification_report.png"))
    
    test_loader = DataLoader(test_tok, batch_size=8)
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
    probs_arr = np.array(all_probs)
    plot_prediction_confidence(probs_arr, true_labels,
                               save_path=os.path.join(RESULTS_DIR, "confidence_distribution.png"))
    
    print_header("ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ BERT")
    model_attn = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
    sample_text = "The cat sat on the mat because it was comfortable."
    visualize_attention(model_attn, tokenizer_orig, sample_text,
                        save_dir=os.path.join(RESULTS_DIR, "attention_plots"))
    
    _, test_raw_samples = prepare_imdb_data(10, 20)
    show_prediction_examples(model, tokenizer_orig, test_raw_samples, num_examples=15)
    
    print_header("ИТОГ")
    print(f"""
[OK] Fill-mask (англ/рус) выполнены.
[OK] Fine-tuning модель загружена из {MODEL_DIR}
   Точность на тесте: {test_acc:.2%}
   F1 = {test_f1:.3f}, Precision = {test_prec:.3f}, Recall = {test_rec:.3f}
[OK] Все графики и таблицы в папке '{RESULTS_DIR}/'
""")

if __name__ == "__main__":
    evaluate()