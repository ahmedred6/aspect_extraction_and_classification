"""Reviewer item: English polarity — macro-F1 alongside accuracy for each model (paper Tables 11-12).

Data: Amazon Review Polarity (fast.ai mirror, 3.6M train / 400k test), downloaded to
      data/raw/amazon_polarity/ (git-ignored; see data/README.md).
Step 1 = src/5_general_polarity/filter_english_amazon_reviews.py (fastText lid.176 keeps English reviews)
         -> paper/notebook sizes: 3,580,535 train / 397,826 test.
Step 2 = src/5_general_polarity/english_polarity.ipynb replayed cell by cell, with macro-F1 recorded next to accuracy.

Paper Table 11 (3-fold CV accuracy): LogReg 0.8747, LinearSVM 0.8753, MultinomialNB 0.8205.
Paper Table 12 (LinearSVM test): accuracy 0.8767, macro-F1 0.88.
"""
import os

import fasttext
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from _common import REPO, save_results

RANDOM_STATE = 42
RAW = os.path.join(REPO, "data", "raw", "amazon_polarity")
OUT = os.path.join(REPO, "experiments", "reviewer_revision", "data")


def filter_english(name, lid):
    """polarity/clearn_amazon_reviews.py filter_english_reviews()."""
    out_path = os.path.join(OUT, f"amazon_{name}_english.csv")
    df = pd.read_csv(os.path.join(RAW, f"{name}.csv"), header=None, names=["label", "title", "text"])
    total = len(df)

    def is_english(text):
        if not isinstance(text, str) or text.strip() == "":
            return False
        (_p, label), = lid.f.predict(text.replace("\n", " ") + "\n", 1, 0.0, "strict")  # == lid.predict()
        return label.replace("__label__", "") == "en"

    df = df[df["text"].apply(lambda x: is_english(str(x)))]
    df.to_csv(out_path, index=False, header=False)
    print(f"{name}: {total} -> {len(df)} English", flush=True)
    return out_path, total, len(df)


def main():
    os.makedirs(OUT, exist_ok=True)
    lid = fasttext.load_model(os.path.join(REPO, "models", "lid.176.ftz"))
    train_path, train_total, train_kept = filter_english("train", lid)
    test_path, test_total, test_kept = filter_english("test", lid)

    # ---- notebook cell 1 ----
    train_df = pd.read_csv(train_path, header=None, names=["label", "title", "text"])
    test_df = pd.read_csv(test_path, header=None, names=["label", "title", "text"])
    n_nan = int(train_df["text"].isna().sum() + test_df["text"].isna().sum())
    X_train, y_train = train_df["text"].fillna(""), train_df["label"]
    X_test, y_test = test_df["text"].fillna(""), test_df["label"]
    label_map = {old: new for new, old in enumerate(sorted(y_train.unique()))}
    y_train, y_test = y_train.map(label_map), y_test.map(label_map)
    print(f"Loaded {len(X_train)} train / {len(X_test)} test; NaN texts: {n_nan}; labels {label_map}", flush=True)

    # ---- notebook cell 2 ----
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ---- notebook cells 3-4 (+ macro-F1) ----
    models = {"LogReg": LogisticRegression(max_iter=2000, n_jobs=-1),
              "LinearSVM": LinearSVC(),
              "MultinomialNB": MultinomialNB()}
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv = {}
    for name, model in models.items():
        accs, f1s = [], []
        for tr, va in kfold.split(X_train_tfidf, y_train):
            preds = model.fit(X_train_tfidf[tr], y_train.iloc[tr]).predict(X_train_tfidf[va])
            accs.append(accuracy_score(y_train.iloc[va], preds))
            f1s.append(f1_score(y_train.iloc[va], preds, average="macro"))
        cv[name] = {"acc_folds": accs, "acc_mean": float(np.mean(accs)),
                    "macro_f1_folds": f1s, "macro_f1_mean": float(np.mean(f1s))}
        print(f"{name:14s} acc={np.round(accs, 4)} mean={np.mean(accs):.4f} | "
              f"macroF1={np.round(f1s, 4)} mean={np.mean(f1s):.4f}", flush=True)

    # ---- notebook cells 5-7 ----
    best = max(cv, key=lambda k: cv[k]["acc_mean"])
    final = {"LogReg": LogisticRegression(max_iter=2000, n_jobs=-1), "LinearSVM": LinearSVC(),
             "MultinomialNB": MultinomialNB()}[best].fit(X_train_tfidf, y_train)
    preds = final.predict(X_test_tfidf)
    rep = classification_report(y_test, preds, output_dict=True)
    test = {"model": best, "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "per_class": {"negative": rep["0"], "positive": rep["1"]}}
    print(f"TEST {best}: acc={test['accuracy']:.4f} macroF1={test['macro_f1']:.4f}", flush=True)

    save_results("e06_english_polarity_macro_f1",
                 {"filter": {"train": [train_total, train_kept], "test": [test_total, test_kept]},
                  "nan_texts": n_nan, "cv3": cv, "test": test},
                 paper={"filter_kept": {"train": 3580535, "test": 397826},
                        "cv3_acc_mean": {"LogReg": 0.8747, "LinearSVM": 0.8753, "MultinomialNB": 0.8205},
                        "test": {"model": "LinearSVM", "accuracy": 0.8767, "macro_f1": 0.88}})


if __name__ == "__main__":
    main()
