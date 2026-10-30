# Java Vulnerability Detection — Group 6

Midterm Group Project 2026, Management of Organizational Data.

**Team:** Bakaev Eldos, Murziyev Adil, Zhazylbekov Zanggar

**Method:** CodeBERT + LoRA fine-tuning (main method). DistilBERT + classification head (alternative method).

The model reads one Java function and says if it is vulnerable or not.
We use the datasets from [rmusab/vul-llm-finetune](https://github.com/rmusab/vul-llm-finetune/tree/main/Datasets)
(Shestov et al., IEEE Access, 2025).

## Results (test set, X1 with P3: 4,545 functions, 117 vulnerable)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **CodeBERT + LoRA** (main) | 0.922 | 0.13 | 0.35 | 0.19 | **0.838** | 0.109 |
| DistilBERT + head (alternative) | 0.901 | 0.11 | 0.39 | 0.17 | 0.814 | 0.119 |
| TF-IDF + Logistic Regression | 0.966 | 0.33 | 0.32 | 0.32 | 0.870 | 0.214 |
| Length of the function only | 0.900 | 0.11 | 0.38 | 0.17 | 0.809 | 0.135 |

Important points:

* Only 2.6% of the test functions are vulnerable. A model that always says "not vulnerable" gets 97.4% accuracy, so accuracy alone is not useful here. We compare models with ROC-AUC, PR-AUC, F1 and recall.
* CodeBERT + LoRA is better than DistilBERT and better than the "length only" score. A simple TF-IDF model is still the best on this split.
* The threshold is chosen on the validation set. With a lower threshold ("triage" mode) CodeBERT finds 94 of 117 vulnerabilities (recall 0.80), but with many false alarms.
* If a security team checks the top 5% of functions ranked by CodeBERT, it finds 28% of all vulnerabilities (5.6 times more than random checking).

All numbers: `results/model_comparison.csv`. Figures: `results/figures/`.

## How to run

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` does everything: downloads the data, ETL, data quality check, training of all models, baselines and evaluation.
It takes about 1.5 hours on an Apple M5 laptop (GPU with MPS). With CUDA it is faster. The pretrained models are downloaded from Hugging Face.

You can also run the steps one by one:

```bash
python src/download_data.py
python src/etl.py
python src/data_quality.py
python src/train.py --model codebert_lora --data with_p3
python src/train.py --model distilbert_head --data with_p3 --max-len 256 --epochs 2
python src/baseline.py --data with_p3
python src/evaluate.py
```

Quick test (2–3 minutes): `python src/train.py --model codebert_lora --data with_p3 --limit 400 --epochs 1 --tag smoke`

The notebook `midterm_group6.ipynb` shows all steps with outputs. By default it loads the saved results and does not train again (`RUN_TRAINING = False`).

## Project structure

```
src/
  download_data.py   download the two datasets and check SHA-256
  etl.py             extract -> validate -> clean -> load into SQLite (data/warehouse.db) + CSV for training
  data_quality.py    data quality before and after ETL, class balance, plots
  train.py           training: codebert_lora (main) or distilbert_head (alternative)
  baseline.py        TF-IDF + logistic regression, and "length only"
  evaluate.py        comparison table, confusion matrices, ROC / PR curves, top-k review
  diagrams.py        data lifecycle, ER model, system architecture
  metrics_utils.py   metrics and threshold selection
  config.py          paths and model settings
results/
  model_comparison.csv, data_quality.json, per_cwe_recall.csv, topk_review.csv
  figures/           all plots and diagrams
  runs/<run_id>/     config.json, train.log, loss_history.csv, metrics.json, predictions
  logs/              ETL log, SHA-256 of data and models, technical issues, full run log
midterm_group6.ipynb
run_all.sh
requirements.txt
```

`data/` and `models/` are not in git (too big). The scripts create them.

## Data pipeline

1. **Extract:** 6 jsonl files (2 dataset versions × train / valid / test), 24,288 rows.
2. **Validate:** label must be 0 or 1, status must match the label, code must not be empty.
3. **Clean:** 11 rows where the same code has both labels are removed. 72 copies of code that was also in another split are removed (data leakage).
4. **Load:** SQLite star schema: `fact_sample` + `dim_function`, `dim_commit`, `dim_cwe`, `dim_dataset`, and the tables `experiment`, `metric`, `prediction` for model results.
5. **Train and evaluate:** the models read only the clean CSV files. Results are written back into the database.

## Training settings

| | CodeBERT + LoRA | DistilBERT + head |
|---|---|---|
| Base model | microsoft/codebert-base (125 M) | distilbert-base-uncased (66 M) |
| Trained parameters | 1.18 M (0.94%), LoRA r=16 on query/value | all 66 M |
| Max tokens | 512 | 256 |
| Batch size / epochs | 8 / 3 | 8 / 2 |
| Learning rate | 3e-4 | 3e-5 |
| Loss | cross-entropy, class weight 5.7 for vulnerable | same |
| Best epoch | by validation PR-AUC | same |
| Threshold | best validation F1 with validation accuracy ≥ 93% | same |

Seed 42 everywhere. Batch 16 × 512 tokens did not fit into 16 GB memory, so we used batch 8 (see `results/logs/technical_issues.txt`).

## Limitations

* Accuracy of CodeBERT is 94.8% on validation but 92.2% on test. The test functions are longer (distribution shift).
* The splits are made by commit, not by project. Code of the same project can be in train and test, and this helps TF-IDF.
* Short training because of the laptop: 3 epochs, 0.94% of the weights.
* Only 117 vulnerable functions in the test set, and the labels come from fix commits, so some labels are noisy.
