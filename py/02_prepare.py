#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


# Paths to the CLEANED TSVs from Step 4
PROC_DIR   = "../data/processed"
TRAIN_CLEAN = f"{PROC_DIR}/drugsComTrain_clean.tsv"
TEST_CLEAN  = f"{PROC_DIR}/drugsComTest_clean.tsv"

assert os.path.exists(TRAIN_CLEAN), f"Missing: {TRAIN_CLEAN} (re-run 01_eda to save cleaned files)"
assert os.path.exists(TEST_CLEAN),  f"Missing: {TEST_CLEAN}  (re-run 01_eda to save cleaned files)"

df_train = pd.read_csv(TRAIN_CLEAN, sep="\t")
df_test  = pd.read_csv(TEST_CLEAN,  sep="\t")

print(f"Train shape: {df_train.shape[0]:,} rows × {df_train.shape[1]} columns")
print(f"Test shape : {df_test.shape[0]:,} rows × {df_test.shape[1]} columns")


# In[ ]:


# Create binary labels from numeric 'rating'
# Mapping: Positive (1) = rating >= 8 ; Negative (0) = rating <= 7
df_train["rating"] = pd.to_numeric(df_train["rating"], errors="coerce")
df_test["rating"]  = pd.to_numeric(df_test["rating"],  errors="coerce")

def map_binary_label(r):
    if pd.isna(r):
        return pd.NA
    return 1 if r >= 8 else 0

df_train["label_bin"] = df_train["rating"].map(map_binary_label)
df_test["label_bin"]  = df_test["rating"].map(map_binary_label)

# quick peek
df_train[["rating", "label_bin"]].head(10)


# In[5]:


# Sanity checks: missing + class balance
print("Missing (train):")
print(df_train[["review_clean", "rating", "label_bin"]].isna().sum(), "\n")

print("Missing (test):")
print(df_test[["review_clean", "rating", "label_bin"]].isna().sum(), "\n")

def show_class_stats(df, name=""):
    counts = df["label_bin"].value_counts(dropna=False).sort_index()
    total = counts.sum()
    print(f"{name} — label_bin counts:")
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0
        print(f"  {k}: {v:,}  ({pct:.1f}%)")
    print()

show_class_stats(df_train, "Train")
show_class_stats(df_test,  "Test")


# In[6]:


#  Save labeled versions (still the CLEANED text)
OUT_TRAIN = f"{PROC_DIR}/drugsComTrain_clean_labeled.tsv"
OUT_TEST  = f"{PROC_DIR}/drugsComTest_clean_labeled.tsv"

cols = ["drugName","condition","review","review_clean","rating","label_bin","date","usefulCount"]

df_train[cols].to_csv(OUT_TRAIN, sep="\t", index=False, encoding="utf-8")
df_test[cols].to_csv(OUT_TEST,  sep="\t", index=False, encoding="utf-8")

print("Saved:")
print(f" - {OUT_TRAIN}")
print(f" - {OUT_TEST}")

