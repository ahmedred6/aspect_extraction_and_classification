"""Reviewer items: Arabic polarity — macro-F1 alongside accuracy; TF-IDF fitting only within training data.

(a) paper protocol (polarity/ARABIC_POLARITY.ipynb, reproduced in r02) with macro-F1 added per model.
(b) leakage-safe protocol: stratified 80/20 split FIRST; TF-IDF inside a Pipeline so it is refit on each
    training fold; 3-fold CV model comparison and GridSearch on the training split only; the test split is
    used once for the final evaluation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from _common import REPO, save_results
from r02_arabic_polarity import normalize_arabic

RS = 42


def models():
    return {"LogReg": LogisticRegression(max_iter=2000, n_jobs=-1),
            "LinearSVM": LinearSVC(),
            "MultinomialNB": MultinomialNB()}


def tfidf():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000)


def main():
    df = pd.read_csv(os.path.join(REPO, "data", "raw", "arabic_100k_reviews.tsv"), sep="\t", names=["label", "text"], header=None)
    df = df[df["label"].isin(["Positive", "Negative"])]
    X = df["text"].apply(normalize_arabic)
    y = df["label"].map({"Negative": 0, "Positive": 1})
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)

    # (a) paper protocol + macro-F1
    X_all = tfidf().fit_transform(X)
    paper_protocol = {}
    for name, model in models().items():
        accs, f1s = [], []
        for tr, va in kfold.split(X_all, y):
            p = model.fit(X_all[tr], y.iloc[tr]).predict(X_all[va])
            accs.append(accuracy_score(y.iloc[va], p)); f1s.append(f1_score(y.iloc[va], p, average="macro"))
        paper_protocol[name] = {"cv_acc_folds": accs, "cv_acc_mean": float(np.mean(accs)),
                                "cv_macro_f1_folds": f1s, "cv_macro_f1_mean": float(np.mean(f1s))}
        print(f"(a) {name:14s} acc={np.mean(accs):.4f} macroF1={np.mean(f1s):.4f}")

    # (b) leakage-safe
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    safe_cv = {}
    for name, model in models().items():
        cv = cross_validate(Pipeline([("tfidf", tfidf()), ("clf", model)]), X_tr, y_tr, cv=kfold,
                            scoring={"acc": "accuracy", "f1": "f1_macro"}, n_jobs=1)
        safe_cv[name] = {"cv_acc_folds": cv["test_acc"].tolist(), "cv_acc_mean": float(cv["test_acc"].mean()),
                         "cv_macro_f1_folds": cv["test_f1"].tolist(), "cv_macro_f1_mean": float(cv["test_f1"].mean())}
        print(f"(b) {name:14s} acc={cv['test_acc'].mean():.4f} macroF1={cv['test_f1'].mean():.4f}")
    best = max(safe_cv, key=lambda k: safe_cv[k]["cv_acc_mean"])
    grids = {"LogReg": {"clf__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
             "LinearSVM": {"clf__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
             "MultinomialNB": {"clf__alpha": [0.1, 0.3, 0.5, 1.0]}}
    grid = GridSearchCV(Pipeline([("tfidf", tfidf()), ("clf", models()[best])]), grids[best],
                        cv=3, scoring="accuracy", n_jobs=1).fit(X_tr, y_tr)
    pred = grid.best_estimator_.predict(X_te)
    rep = classification_report(y_te, pred, output_dict=True)
    test = {"model": best, "best_params": grid.best_params_, "n_train": len(y_tr), "n_test": len(y_te),
            "accuracy": accuracy_score(y_te, pred), "macro_f1": f1_score(y_te, pred, average="macro"),
            "per_class": {"negative": rep["0"], "positive": rep["1"]}}
    print(f"(b) TEST {best} {grid.best_params_} acc={test['accuracy']:.4f} macroF1={test['macro_f1']:.4f}")

    save_results("e02_arabic_polarity_macro_f1",
                 {"a_paper_protocol_cv": paper_protocol, "b_leakage_safe_cv": safe_cv, "b_leakage_safe_test": test},
                 paper={"cv_acc_mean": {"LogReg": 0.8679, "LinearSVM": 0.8614, "MultinomialNB": 0.8582},
                        "test_accuracy": 0.8742, "test_macro_f1": 0.87})


if __name__ == "__main__":
    main()
