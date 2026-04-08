# BERT Demo: Fill‑mask и Fine‑tuning

Демонстрация работы модели BERT для презентации по статье  
*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (arXiv:1810.04805).

## Структура проекта

- `config.py` – все параметры (размер выборки, эпохи, пути, seed).  
- `utils.py` – общие функции (fill‑mask, графики, таблицы, визуализация внимания).  
- `train.py` – загрузка данных, обучение модели и сохранение весов в `models/bert_imdb_finetuned`.  
- `evaluate.py` – загрузка сохранённой модели и генерация всех графиков/таблиц без повторного обучения.  
- `bert_demo_results/` – папка с результатами (картинки, CSV).  
- `bert_training_cache/` – временные чекпоинты (игнорируется Git).  

## Результаты

- **Fill‑mask (англ/рус)**: BERT предсказывает пропущенные слова, учитывая контекст.  
  Примеры: `sitting on [MASK]` → `floor` (33.7%), `drinking [MASK]` → `water` (55.1%).  
- **Fine‑tuning на IMDb** (1000 трейн, 300 тест):  
  - Baseline (случайно): 50%  
  - Точность после дообучения: **~85.7%**  
  - F1‑score: 0.86, Precision: 0.84, Recall: 0.88  

Все графики и таблицы доступны в папке [`bert_demo_results/`](bert_demo_results).

## Запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

Или вручную:

```bash
pip install transformers datasets torch matplotlib seaborn pandas scikit-learn
```

### 2. Обучение модели (один раз)
```bash
python train.py
```
После завершения модель сохранится в models/bert_imdb_finetuned.

### 3. Генерация всех графиков и таблиц
```bash
python evaluate.py
```
Загрузит сохранённую модель и создаст fill‑mask, матрицу ошибок, кривые обучения, примеры предсказаний и т.д. в папке bert_demo_results.

## Примечания
Тяжёлые папки (bert_training_cache/, models/) добавлены в .gitignore и не попадают в репозиторий.

Если нужно изменить параметры (размер выборки, число эпох), отредактируйте config.py и запустите train.py заново.

Для просмотра графиков откройте папку bert_demo_results.

### Ссылки
[Оригинальная статья BERT](https://arxiv.org/abs/1810.04805)