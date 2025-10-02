# Natural Language Processing: Drug Review Sentiment 

Predict **positive vs negative** sentiment from **patient free‑text** medication reviews (Drugs.com). Labels are derived from the 1–10 numeric ratings; models use only the review text.

---

## Overview 

* Map ratings to **binary sentiment** (positive / negative).
* Train and evaluate two approaches:

  * **Baseline**: TF‑IDF + Logistic Regression.
  * **Transformer**: DistilBERT fine‑tune.
* Report accuracy and **macro‑F1**, plus a confusion matrix.

## Dataset

* **Name**: Drug Review Dataset (Drugs.com)
* **Files**: `drugsComTrain_raw.tsv`, `drugsComTest_raw.tsv`
* **Columns**: `drugName`, `condition`, `review`, `rating` (1–10), `date`, `usefulCount`
* **Access**: UCI Machine Learning Repository (public)

  * Zip bundle: `drugsCom_raw.zip`
  * URL: [https://archive.ics.uci.edu/ml/machine-learning-databases/00462/](https://archive.ics.uci.edu/ml/machine-learning-databases/00462/)

### Download via shell (Linux)

From the project root:

```bash
mkdir -p drug-review-sentiment/data
cd data

# download the UCI Drugs.com Reviews bundle
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip -O drugsCom_raw.zip

unzip -o drugsCom_raw.zip

```

# Drug Review Sentiment (Binary)

Predict **positive vs negative** sentiment from **patient free‑text** medication reviews (Drugs.com). Labels are derived from 1–10 ratings; models use only the review text.

---

## Overview

* Load the public Drugs.com reviews dataset.
* Create a minimally cleaned text column (`review_clean`).
* Map ratings to **binary sentiment** (positive / negative).
* Train and evaluate:

  * **Baseline:** TF‑IDF + Logistic Regression.
  * **Transformer:** DistilBERT fine‑tune.
* Report Accuracy and **Macro‑F1**, plus confusion matrices.
* Device‑agnostic notebooks (CPU or GPU), though transformer benefits from GPU.

---

## Dataset

* **Name:** Drug Review Dataset (Drugs.com)
* **Files:** `drugsComTrain_raw.tsv`, `drugsComTest_raw.tsv`
* **Columns:** `drugName`, `condition`, `review`, `rating` (1–10), `date`, `usefulCount`
* **Source:** UCI Machine Learning Repository (public):

  * Base URL: [https://archive.ics.uci.edu/ml/machine-learning-databases/00462/](https://archive.ics.uci.edu/ml/machine-learning-databases/00462/)
  * Bundle: `drugsCom_raw.zip`

### Download (Linux shell)

From the project root:

```bash
# create data folder and move into it
mkdir -p data
cd data

# download the UCI bundle
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip -O drugsCom_raw.zip

# unzip to get the two TSVs:
#   - drugsComTrain_raw.tsv
#   - drugsComTest_raw.tsv
unzip -o drugsCom_raw.zip

# quick sanity checks (optional)
ls -lh drugsCom*raw.tsv
wc -l drugsComTrain_raw.tsv drugsComTest_raw.tsv

# return to project root
cd ..
```

---

## Labeling (Binary)

* **Mapping used:**

  * **Positive** → `rating ≥ 8`
  * **Negative** → `rating ≤ 7`
* **Notes:**

  * Ratings are skewed high; report **Macro‑F1** alongside accuracy.
  * Mapping is configurable in later scripts if needed.

---

## Text Cleaning

Create `review_clean` with minimal normalization:

1. **HTML entity unescape** (e.g., `&#039;` → `'`, `&quot;` → `"`).
2. **Whitespace normalization** (collapse runs; trim ends).
3. Preserve **casing & punctuation** (useful for sentiment).

The original `review` column is retained for traceability.

---

## Repository Structure

```
drug-review-sentiment/
├── artifacts/            # models, metrics (ignored by Git; placeholders kept)
├── data/                 # raw TSVs (ignored by Git)
│   └── processed/        # cleaned/processed TSVs
├── notebooks/
│   ├── 01_eda.ipynb              # load raw, clean text, light EDA
│   ├── 02_prepare.ipynb          # add binary labels; save processed TSVs
│   ├── 03_baseline_tfidf_lr.ipynb# TF‑IDF + Logistic Regression
│   └── 04_transformer_distilbert.ipynb # DistilBERT fine‑tune
├── reports/
│   └── figures/          # confusion matrices, plots
├── src/                  # (optional) helper/train scripts later
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quick Start

1. **Download** the dataset (shell snippet above) into `data/`.
2. Run notebooks in order:

   * `01_eda.ipynb` → create `review_clean`; (optionally) save cleaned TSVs.
   * `02_prepare.ipynb` → add `label_bin` and save `*_clean_labeled.tsv` in `data/processed/`.
   * `03_baseline_tfidf_lr.ipynb` → train TF‑IDF + Logistic Regression; save metrics and confusion matrix.
   * `04_transformer_distilbert.ipynb` → fine‑tune DistilBERT; save test metrics, confusion matrix, and model/tokenizer.

> `data/` and `artifacts/` contents are Git‑ignored; only small figures/metadata are versioned.

---

## Results

| Model                   | Accuracy | Macro‑F1 | F1 (neg) | F1 (pos) |
| ----------------------- | :------: | :------: | :------: | :------: |
| TF‑IDF + Logistic Reg   |  0.8642  |  0.8597  |  0.8343  |  0.8850  |
| DistilBERT (fine‑tuned) |  0.9055  |  0.9010  |  0.8799  |  0.9221  |

**Figures (paths):**

* `reports/figures/distilbert_confusion_matrix.png`
* `reports/figures/tfidf_confusion_matrix.png`

**DistilBERT notes:** 2 epochs, `max_length=256`, batch size 16, mixed precision on CUDA; metrics are from the held‑out test split.

---

## Reproduce (concise)

* `01_eda.ipynb` → clean text
* `02_prepare.ipynb` → label mapping → `*_clean_labeled.tsv`
* `03_baseline_tfidf_lr.ipynb` → TF‑IDF + LR → metrics + figure
* `04_transformer_distilbert.ipynb` → fine‑tune → test metrics + figure + saved model

---

## Ethics & Limitations

* Ratings are self‑reported and skew positive; not clinical ground truth.
* Label mapping is a design choice and affects class balance.
* **No clinical use.** Research/education only.

---

## Attribution

* Dataset: **Drug Review Dataset (Drugs.com)** — UCI Machine Learning Repository.

---

## License

* Project license: **TBD** (e.g., MIT). Add a LICENSE file before release.


