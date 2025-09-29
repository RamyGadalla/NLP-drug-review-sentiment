# NLP: Drug Review Sentiment (under development)

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

---

---

##
