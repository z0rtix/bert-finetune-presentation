# BERT Demo: Fill‑mask и Fine‑tuning

Демонстрация работы модели BERT для презентации по статье  
*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (arXiv:1810.04805).

## Результаты

- **Fill‑mask (англ/рус)**: BERT предсказывает пропущенные слова, учитывая контекст (например, `floor` для сидящей кошки, `water` — для пьющей).  
- **Fine‑tuning на IMDb**:  
  - Baseline (случайно): 50%  
  - Точность после дообучения: **~85.7%**  
  - F1-score: ~0.86, Precision/Recall сбалансированы.

## Графики и таблицы

Все файлы в папке [`bert_demo_results/`](bert_demo_results):
- `fillmask_en.csv`, `fillmask_ru.csv` — примеры предсказаний с вероятностями.  
- `accuracy_comparison.png` — сравнение со случайным угадыванием.  
- `training_curves.png` — динамика loss и accuracy.  
- `confusion_matrix.png` — матрица ошибок.  
- `classification_report.png` — детальные метрики.  
- `confidence_distribution.png` — уверенность модели.  
- `attention_plots/` — визуализация внимания BERT.  
- `predictions_examples.csv` — живые примеры предсказаний на отзывах.

## Запуск

```bash
pip install transformers datasets torch matplotlib seaborn pandas scikit-learn
python bert_demo_advanced.py