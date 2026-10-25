"""Reviewer item: macro-F1 alongside accuracy for the Decision Tree and Linear SVM rows of Table 8.

The script that produced those two rows is not in the repo, so they are rebuilt following the method the
paper describes in §6.7:
  - same data, 80/20 stratified split (seed 42) and test set as the Stacking row (r01 / r03)
  - vectorization per §6.7.1: English = word TF-IDF 1-3 grams; Arabic = word 1-3 + char 3-5 TF-IDF
    (same TF-IDF parameters as the repo's get_advanced_vectorizer)
  - SMOTE on the training split (§6.7.2)
Main hyperparameters are swept; for each row the run whose accuracy is closest to Table 8 is reported with
its macro-F1, plus the macro-F1 spread over all runs within +-1 accuracy point of Table 8.
"""
import os
from itertools import product

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from _common import REPO, load_original, save_results

ar = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")
en = load_original("src/4_aspect_sentiment/aspect_sentiment_english.py", "AspClassEnglish")
RS = 42
PAPER = {"english": {"Decision Tree": 0.765, "Linear SVM": 0.782},
         "arabic": {"Decision Tree": 0.712, "Linear SVM": 0.855}}

DT_GRID = [dict(criterion=c, max_depth=d, min_samples_leaf=l)
           for c, d, l in product(["gini", "entropy"], [5, 10, 15, 20, 30, 50, None], [1, 5, 20])]
SVM_GRID = [dict(C=c) for c in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]]


def split(lang):
    if lang == "arabic":
        mod, df = ar, ar.load_data(os.path.join(REPO, ar.DATA_PATH))
    else:
        mod, df = en, en.load_data(os.path.join(REPO, "repro", "data", "cleaned_final_dataset.jsonl"))
    df = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    return mod, train_test_split(df["text"], df["label"], test_size=mod.TEST_SIZE,
                                 stratify=df["label"], random_state=mod.RANDOM_STATE)


def vectorizer(lang, mod):
    if lang == "arabic":
        return mod.get_advanced_vectorizer()  # word 1-3 + char_wb 3-5
    return TfidfVectorizer(ngram_range=(1, 3), max_features=20000, sublinear_tf=True,
                           analyzer="word", token_pattern=r"\w{1,}")  # §6.7.1: English word 1-3 only


def run(family, params, Xtr, ytr, Xte, yte):
    clf = (DecisionTreeClassifier(random_state=RS, **params) if family == "Decision Tree"
           else LinearSVC(class_weight="balanced", random_state=RS, max_iter=5000, **params))
    p = ImbPipeline([("smote", SMOTE(random_state=RS, k_neighbors=5)), ("clf", clf)]).fit(Xtr, ytr).predict(Xte)
    return {"family": family, "params": params, "accuracy": accuracy_score(yte, p),
            "macro_f1": f1_score(yte, p, average="macro"),
            "confusion_matrix[neg,pos]": confusion_matrix(yte, p, labels=["negative", "positive"]).tolist()}


def main():
    out = {}
    for lang in ("english", "arabic"):
        mod, (X_train, X_test, y_train, y_test) = split(lang)
        vec = vectorizer(lang, mod)
        Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
        jobs = [("Decision Tree", p) for p in DT_GRID] + [("Linear SVM", p) for p in SVM_GRID]
        runs = Parallel(n_jobs=-1)(delayed(run)(f, p, Xtr, y_train, Xte, y_test) for f, p in jobs)
        out[lang] = {"n_test": len(y_test), "runs": runs, "rows": {}}
        print(f"\n######## {lang} (test n={len(y_test)})")
        for family in ("Decision Tree", "Linear SVM"):
            target = PAPER[lang][family]
            fam = [r for r in runs if r["family"] == family]
            best = min(fam, key=lambda r: abs(r["accuracy"] - target))
            near = [r["macro_f1"] for r in fam if abs(r["accuracy"] - target) <= 0.01]
            out[lang]["rows"][family] = {
                "paper_accuracy": target, "closest_run": best,
                "macro_f1_range_within_1pt": [min(near), max(near)] if near else None,
                "n_runs_within_1pt": len(near)}
            print(f"{family:14s} paper acc={target:.3f} | closest run acc={best['accuracy']:.4f} "
                  f"macroF1={best['macro_f1']:.4f} params={best['params']} | "
                  f"runs within ±1pt: {len(near)}, macroF1 {('%.3f-%.3f' % (min(near), max(near))) if near else 'n/a'}")
    save_results("e05_table8_dt_svm_macro_f1", out, paper={"table8_accuracy": PAPER})


if __name__ == "__main__":
    main()
