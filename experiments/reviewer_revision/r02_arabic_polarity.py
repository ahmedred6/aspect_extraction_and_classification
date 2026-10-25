"""Reproduce paper §6.8.2 / Tables 13-14 — Arabic general polarity.

Original: polarity/ARABIC_POLARITY.ipynb (cells replicated verbatim below)
Data:     ar_reviews_100k.tsv (Arabic 100K Reviews; Mixed removed -> 66,666)

Paper: 3-fold CV LogReg 0.8679 / LinearSVM 0.8614 / MNB 0.8582;
       final LogReg test accuracy 0.8742 on 13,334 reviews.
NOTE (leakage, as in the original): TF-IDF and GridSearch are fit on ALL 66,666
rows before the 80/20 split, so the "test" rows were seen during fitting.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from _common import REPO, save_results

RANDOM_STATE = 42


def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[ؗ-ًؚ-ْ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r"[ىي]", "ي", text)
    return text


def main():
    df = pd.read_csv(os.path.join(REPO, "data", "raw", "arabic_100k_reviews.tsv"), sep="\t", names=["label", "text"], header=None)
    df = df[df["label"].isin(["Positive", "Negative", "Mixed"])]
    df = df[df["label"] != "Mixed"]
    counts = df["label"].value_counts().to_dict()
    print(counts)
    df["clean_text"] = df["text"].apply(normalize_arabic)
    X, y = df["clean_text"], df["label"]
    y = y.map({old: new for new, old in enumerate(sorted(y.unique()))})

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000)
    X_tfidf = vectorizer.fit_transform(X)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=-1),
        "LinearSVM": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
    }
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv = {}
    for name, model in models.items():
        fold_acc = []
        for tr, va in kfold.split(X_tfidf, y):
            model.fit(X_tfidf[tr], y.iloc[tr])
            fold_acc.append(accuracy_score(y.iloc[va], model.predict(X_tfidf[va])))
        cv[name] = {"folds": fold_acc, "mean": float(np.mean(fold_acc))}
        print(f"{name}: {np.round(fold_acc, 4)} mean={np.mean(fold_acc):.4f}")

    grid = GridSearchCV(LogisticRegression(max_iter=2000, n_jobs=-1),
                        {"C": [0.1, 0.5, 1.0, 2.0, 5.0], "penalty": ["l2"]},
                        cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_tfidf, y)
    final_model = grid.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    print(f"best params {grid.best_params_}  TEST acc={acc:.4f}  macro-F1={mf1:.4f}  n_test={len(y_test)}")

    save_results(
        "r02_arabic_polarity",
        {"class_counts": counts, "cv": cv, "best_params": grid.best_params_,
         "test_accuracy": acc, "macro_f1": mf1, "n_test": int(len(y_test))},
        paper={"cv_mean": {"LogReg": 0.8679, "LinearSVM": 0.8614, "MultinomialNB": 0.8582},
               "test_accuracy": 0.8742, "n_test": 13334},
    )


if __name__ == "__main__":
    main()
