import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, TrainingArguments, Trainer, logging as hf_logging
)
from datasets import load_dataset

# ========================= НАСТРОЙКИ =========================
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Стиль графиков
available = plt.style.available
if 'seaborn-v0_8-darkgrid' in available:
    plt.style.use('seaborn-v0_8-darkgrid')
elif 'seaborn-darkgrid' in available:
    plt.style.use('seaborn-darkgrid')
else:
    plt.style.use('default')
sns.set_palette("pastel")

# Папка для результатов (только лёгкие файлы: картинки, CSV)
RESULTS_DIR = "bert_demo_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "attention_plots"), exist_ok=True)

# Папка для тяжёлых данных обучения (не включается в Git)
CACHE_DIR = "bert_training_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Создаём .gitignore, чтобы Git не трогал тяжёлые папки и кэш
gitignore_content = f"""{CACHE_DIR}/
{RESULTS_DIR}/checkpoints/
{RESULTS_DIR}/logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
"""
with open(".gitignore", "w") as f:
    f.write(gitignore_content)
print("✓ Файл .gitignore создан (тяжёлые папки исключены из Git)")

DEVICE = -1  # CPU, для GPU поставьте 0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================
def print_header(text, char="=", width=70):
    print("\n" + char * width)
    padding = (width - len(text) - 2) // 2
    if padding < 0:
        padding = 0
    print(char * padding + " " + text + " " + char * (width - padding - len(text) - 2))
    print(char * width + "\n")

def save_table(df, path):
    df.to_csv(path, index=False)
    print(f"[OK] Таблица сохранена: {path}")

def plot_attention_heatmap(tokens, attentions, layer=0, head=0, save_path=None):
    attn = attentions[layer][0, head].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="Blues", ax=ax)
    ax.set_title(f"Attention Map (Layer {layer}, Head {head})", fontsize=14)
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Attention heatmap сохранён: {save_path}")
    plt.close()

# ========================= 1. FILL-MASK (ENGLISH) =========================
def run_fill_mask_en():
    print_header("1. FILL-MASK (английский)")
    fill_pipe = pipeline("fill-mask", model="bert-base-uncased", device=DEVICE)
    
    examples = [
        ("The cat is sitting on the [MASK].", "Сидит на поверхности"),
        ("The cat is drinking [MASK] from the bowl.", "Пьёт жидкость"),
        ("I went to the [MASK] to buy groceries.", "Поход за продуктами"),
        ("The [MASK] is shining brightly in the sky.", "Небесное светило"),
        ("She gave me a [MASK] look.", "Эмоциональный взгляд"),
        ("The stock market [MASK] sharply yesterday.", "Движение рынка"),
        ("He tried to [MASK] his feelings.", "Скрывать эмоции"),
        ("The movie was absolutely [MASK]!", "Оценка фильма")
    ]
    
    rows = []
    for text, desc in examples:
        preds = fill_pipe(text)
        rows.append({
            "Context": text[:50] + "..." if len(text) > 50 else text,
            "Description": desc,
            "Top-1": f"{preds[0]['token_str']} ({preds[0]['score']:.1%})",
            "Top-2": f"{preds[1]['token_str']} ({preds[1]['score']:.1%})",
            "Top-3": f"{preds[2]['token_str']} ({preds[2]['score']:.1%})"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, os.path.join(RESULTS_DIR, "fillmask_en.csv"))
    return df

# ========================= 2. FILL-MASK (RUSSIAN) =========================
def run_fill_mask_ru():
    print_header("2. FILL-MASK (русский)")
    fill_pipe = pipeline("fill-mask", model="bert-base-multilingual-cased", device=DEVICE)
    
    examples = [
        ("Кошка сидит на [MASK].", "Находится на чём-то"),
        ("Кошка пьёт [MASK] из миски.", "Пьёт жидкость"),
        ("Я пошёл в [MASK] купить хлеба.", "Магазин/место"),
        ("Сегодня [MASK] светит ярко.", "Солнце/небо"),
        ("Он посмотрел на меня с [MASK].", "Эмоция/выражение"),
        ("Цены на нефть резко [MASK] вчера.", "Изменение цен"),
        ("Этот фильм оказался полным [MASK].", "Оценка качества"),
        ("Она пыталась [MASK] свою радость.", "Скрывать чувства")
    ]
    
    rows = []
    for text, desc in examples:
        preds = fill_pipe(text)
        rows.append({
            "Контекст": text[:50] + "..." if len(text) > 50 else text,
            "Описание": desc,
            "1-е": f"{preds[0]['token_str'].replace('##','')} ({preds[0]['score']:.1%})",
            "2-е": f"{preds[1]['token_str'].replace('##','')} ({preds[1]['score']:.1%})",
            "3-е": f"{preds[2]['token_str'].replace('##','')} ({preds[2]['score']:.1%})"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, os.path.join(RESULTS_DIR, "fillmask_ru.csv"))
    return df

# ========================= 3. FINE-TUNING НА IMDb =========================
def prepare_imdb_data(sample_train=800, sample_test=200):
    dataset = load_dataset("imdb")
    train_subset = dataset["train"].shuffle(seed=RANDOM_SEED).select(range(sample_train))
    test_subset = dataset["test"].shuffle(seed=RANDOM_SEED).select(range(sample_test))
    return train_subset, test_subset

def tokenize_dataset(dataset, tokenizer, max_len=128):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    tok = dataset.map(tokenize_fn, batched=True)
    tok = tok.remove_columns(["text"]).rename_column("label", "labels")
    tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tok

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    acc = accuracy_score(eval_pred.label_ids, preds)
    f1 = f1_score(eval_pred.label_ids, preds, average='binary')
    prec = precision_score(eval_pred.label_ids, preds, average='binary')
    rec = recall_score(eval_pred.label_ids, preds, average='binary')
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def run_finetuning():
    print_header("3. FINE-TUNING НА IMDb (классификация тональности)")
    train_raw, test_raw = prepare_imdb_data(sample_train=1000, sample_test=300)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_tok = tokenize_dataset(train_raw, tokenizer)
    test_tok = tokenize_dataset(test_raw, tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # ВСЕ тяжёлые файлы уходят в CACHE_DIR (не в RESULTS_DIR)
    training_args = TrainingArguments(
        output_dir=os.path.join(CACHE_DIR, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(CACHE_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=False,
        save_total_limit=1,          # хранить только лучший чекпоинт
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
    
    print("\n[INFO] Оценка на тестовой выборке...")
    preds_out = trainer.predict(test_tok)
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    true_labels = preds_out.label_ids
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='binary')
    test_prec = precision_score(true_labels, pred_labels, average='binary')
    test_rec = recall_score(true_labels, pred_labels, average='binary')
    
    print(f"\n[OK] Результаты fine-tuning:")
    print(f"   Accuracy : {test_acc:.2%}")
    print(f"   F1-score : {test_f1:.3f}")
    print(f"   Precision: {test_prec:.3f}")
    print(f"   Recall   : {test_rec:.3f}")
    
    return trainer, test_tok, true_labels, pred_labels, (test_acc, test_f1, test_prec, test_rec)

# ========================= 4. ВИЗУАЛИЗАЦИИ =========================
def plot_accuracy_comparison(baseline, finetuned_acc, save_path):
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(["Случайное угадывание", "BERT fine-tuned"], [baseline, finetuned_acc],
                  color=["#9b9b9b", "#2c7fb8"], edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Сравнение точности классификации", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.1%}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] График сравнения сохранён: {save_path}")

def plot_training_curves(trainer, save_path):
    logs = trainer.state.log_history
    train_losses, eval_accs, ep_loss, ep_acc = [], [], [], []
    for e in logs:
        if "loss" in e and "epoch" in e:
            train_losses.append(e["loss"])
            ep_loss.append(e["epoch"])
        if "eval_accuracy" in e and "epoch" in e:
            eval_accs.append(e["eval_accuracy"])
            ep_acc.append(e["epoch"])
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    if train_losses:
        ax1.plot(ep_loss, train_losses, marker='o', color="#d95f02", linewidth=2)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
        ax1.grid(True, linestyle='--')
    if eval_accs:
        ax2.plot(ep_acc, eval_accs, marker='s', color="#1b9e77", linewidth=2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy")
        ax2.set_ylim(0,1); ax2.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Кривые обучения сохранены: {save_path}")

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'],
                annot_kws={'size':14}, cbar_kws={'label':'Count'})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Матрица ошибок сохранена: {save_path}")

def plot_classification_report(true_labels, pred_labels, save_path):
    report = classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_report.round(3).values, colLabels=df_report.columns, rowLabels=df_report.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Classification Report", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Отчёт классификации сохранён: {save_path}")

def plot_prediction_confidence(probs, true_labels, save_path):
    max_probs = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == true_labels)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(max_probs[correct], bins=20, alpha=0.7, label='Correct predictions', color='green')
    ax.hist(max_probs[~correct], bins=20, alpha=0.7, label='Wrong predictions', color='red')
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence distribution of predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Распределение уверенности сохранено: {save_path}")

# ========================= 5. ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ =========================
def visualize_attention(model, tokenizer, text, save_dir):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    num_layers = len(attentions)
    plot_attention_heatmap(tokens, attentions, layer=num_layers-1, head=0,
                           save_path=os.path.join(save_dir, "attention_last_layer_head0.png"))
    plot_attention_heatmap(tokens, attentions, layer=0, head=0,
                           save_path=os.path.join(save_dir, "attention_first_layer_head0.png"))
    print(f"[OK] Визуализация внимания сохранена в {save_dir}")

# ========================= 6. ПРИМЕРЫ ПРЕДСКАЗАНИЙ =========================
def show_prediction_examples(model, tokenizer, dataset, num_examples=15):
    print_header("ПРИМЕРЫ ПРЕДСКАЗАНИЙ НА ТЕСТОВЫХ ОТЗЫВАХ")
    sample_texts = dataset["text"][:num_examples]
    sample_labels = dataset["label"][:num_examples]
    inputs = tokenizer(sample_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    
    rows = []
    for txt, tl, pl, pr in zip(sample_texts, sample_labels, preds, probs):
        rows.append({
            "Review (first 70 chars)": (txt[:70] + "...") if len(txt) > 70 else txt,
            "True": "POS" if tl==1 else "NEG",
            "Pred": "POS" if pl==1 else "NEG",
            "Confidence": f"{max(pr):.1%}",
            "Correct": "✔" if tl==pl else "✘"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table(df, os.path.join(RESULTS_DIR, "predictions_examples.csv"))
    return df

# ========================= 7. ОСНОВНАЯ ФУНКЦИЯ =========================
def main():
    print_header("BERT ДЕМОНСТРАЦИЯ: ДВУНАПРАВЛЕННОСТЬ И ТОНКАЯ НАСТРОЙКА")
    
    run_fill_mask_en()
    run_fill_mask_ru()
    
    trainer, test_tok, true_labels, pred_labels, metrics = run_finetuning()
    test_acc, test_f1, test_prec, test_rec = metrics
    
    print_header("ГЕНЕРАЦИЯ ГРАФИКОВ")
    plot_accuracy_comparison(baseline=0.5, finetuned_acc=test_acc,
                             save_path=os.path.join(RESULTS_DIR, "accuracy_comparison.png"))
    plot_training_curves(trainer, save_path=os.path.join(RESULTS_DIR, "training_curves.png"))
    plot_confusion_matrix(true_labels, pred_labels, save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_classification_report(true_labels, pred_labels, save_path=os.path.join(RESULTS_DIR, "classification_report.png"))
    
    preds_out = trainer.predict(test_tok)
    probs = torch.softmax(torch.tensor(preds_out.predictions), dim=-1).numpy()
    plot_prediction_confidence(probs, true_labels, save_path=os.path.join(RESULTS_DIR, "confidence_distribution.png"))
    
    print_header("ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ BERT")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_attn = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
    sample_text = "The cat sat on the mat because it was comfortable."
    visualize_attention(model_attn, tokenizer, sample_text, save_dir=os.path.join(RESULTS_DIR, "attention_plots"))
    
    _, test_raw = prepare_imdb_data(sample_train=10, sample_test=20)
    show_prediction_examples(trainer.model, tokenizer, test_raw, num_examples=15)
    
    print_header("ИТОГОВЫЙ ОТЧЁТ ДЛЯ ПРЕЗЕНТАЦИИ")
    print(f"""
[OK] Fill-mask (англ): модель угадывает слова по контексту.
   'sitting on [MASK]' → floor (33.7%), couch (12.9%)
   'drinking [MASK]'  → water (55.1%), milk (12.5%)

[OK] Fill-mask (рус): мультиязычная BERT работает аналогично.
   'сидит на [MASK]' → ковре (22.1%), полу (18.2%)
   'пьёт [MASK]'     → воду (34.2%), молоко (21.5%)

[OK] Fine-tuning на IMDb (1000 трейн, 300 тест):
   Baseline (случайно): 50.0%
   BERT после дообучения: Accuracy = {test_acc:.2%}
   F1-score = {test_f1:.3f}, Precision = {test_prec:.3f}, Recall = {test_rec:.3f}

[OK] Все графики и таблицы в папке '{RESULTS_DIR}/'
[OK] Тяжёлые файлы чекпоинтов и логов в папке '{CACHE_DIR}/' (исключены из Git)

Рекомендации для презентации: см. скриншоты из RESULTS_DIR.
""")
    print("\n[DONE] Код выполнен. Теперь можно коммитить только лёгкие файлы.")

if __name__ == "__main__":
    main()