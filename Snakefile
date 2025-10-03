configfile: "config.yaml"

DATA        = config.get("data_dir", "data")
PROCESSED   = config.get("processed_dir", "data/processed")
ARTIFACTS   = config.get("artifacts_dir", "artifacts")
FIGURES     = config.get("figures_dir", "reports/figures")
PY_DIR      = config.get("py_dir", "py")
RUN_TFIDF   = bool(config.get("run", {}).get("tfidf", True))
RUN_DISTIL  = bool(config.get("run", {}).get("distilbert", True))

RAW_TRAIN   = f"{DATA}/drugsComTrain_raw.tsv"
RAW_TEST    = f"{DATA}/drugsComTest_raw.tsv"

CLEAN_TRAIN = f"{PROCESSED}/drugsComTrain_clean.tsv"
CLEAN_TEST  = f"{PROCESSED}/drugsComTest_clean.tsv"

LAB_TRAIN   = f"{PROCESSED}/drugsComTrain_clean_labeled.tsv"
LAB_TEST    = f"{PROCESSED}/drugsComTest_clean_labeled.tsv"

TFIDF_METR  = f"{ARTIFACTS}/tfidf_baseline/metrics.json"
TFIDF_FIG   = f"{FIGURES}/tfidf_confusion_matrix.png"

DISTIL_METR = f"{ARTIFACTS}/distilbert_binary/metrics_test.json"
DISTIL_FIG  = f"{FIGURES}/distilbert_confusion_matrix.png"

# ---- Targets ----
_targets = [LAB_TRAIN, LAB_TEST]
if RUN_TFIDF:
    _targets += [TFIDF_METR, TFIDF_FIG]
if RUN_DISTIL:
    _targets += [DISTIL_METR, DISTIL_FIG]

rule all:
    input: _targets

# 1) Download raw TSVs from UCI
rule download:
    output:
        RAW_TRAIN,
        RAW_TEST
    params:
        data_dir = DATA
    shell:
        r"""
        mkdir -p {params.data_dir}
        cd {params.data_dir}
        wget -q -O drugsCom_raw.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
        unzip -o drugsCom_raw.zip
        """

# 2) Clean text → write *_clean.tsv (relies on your exported 01_eda.py)
rule clean:
    input:
        RAW_TRAIN,
        RAW_TEST
    output:
        CLEAN_TRAIN,
        CLEAN_TEST
    params:
        py_dir   = PY_DIR,
        out_dir  = PROCESSED
    shell:
        r"""
        mkdir -p {params.out_dir}
        python {params.py_dir}/01_eda.py
        """

# 3) Label mapping → write *_clean_labeled.tsv (uses 02_prepare.py)
rule label:
    input:
        CLEAN_TRAIN,
        CLEAN_TEST
    output:
        LAB_TRAIN,
        LAB_TEST
    params:
        py_dir = PY_DIR
    shell:
        r"""
        python {params.py_dir}/02_prepare.py
        """

# 4) Baseline model (TF‑IDF + LR) → metrics + confusion matrix
rule tfidf_train:
    input:
        LAB_TRAIN,
        LAB_TEST
    output:
        TFIDF_METR,
        TFIDF_FIG
    params:
        py_dir = PY_DIR
    shell:
        r"""
        python {params.py_dir}/03_baseline_tfidf_lr.py
        """

# 5) DistilBERT fine‑tune → test metrics + confusion matrix
rule distilbert_train:
    input:
        LAB_TRAIN,
        LAB_TEST
    output:
        DISTIL_METR,
        DISTIL_FIG
    params:
        py_dir = PY_DIR,
        cuda   = config.get("distilbert", {}).get("cuda_visible_devices", "")
    shell:
        r"""
        export CUDA_VISIBLE_DEVICES="{params.cuda}"
        python {params.py_dir}/04_transformer_distilbert.py
        """