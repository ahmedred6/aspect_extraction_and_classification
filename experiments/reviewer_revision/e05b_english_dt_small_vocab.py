"""Follow-up to e05: English Decision Tree row of Table 8 with smaller vocabularies.

Same split / SMOTE / word 1-3 TF-IDF as e05; only max_features (and binary counts) vary, because a tree on
20k sparse n-grams overfits (e05 best: 64.9%). Reports the run closest to Table 8's 76.5%.
"""
from itertools import product

from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from _common import save_results
from e05_table8_dt_svm_macro_f1 import run, split

TARGET = 0.765


def job(kind, max_features, params, X_train, X_test, y_train, y_test):
    vec = (TfidfVectorizer(ngram_range=(1, 3), max_features=max_features, sublinear_tf=True, token_pattern=r"\w{1,}")
           if kind == "tfidf" else
           CountVectorizer(ngram_range=(1, 3), max_features=max_features, binary=True, token_pattern=r"\w{1,}"))
    r = run("Decision Tree", params, vec.fit_transform(X_train), y_train, vec.transform(X_test), y_test)
    r.update({"features": kind, "max_features": max_features})
    return r


def main():
    _, (X_train, X_test, y_train, y_test) = split("english")
    grid = [dict(criterion=c, max_depth=d, min_samples_leaf=l)
            for c, d, l in product(["gini", "entropy"], [10, 20, 40, None], [1, 5, 20])]
    jobs = [(k, mf, p) for k in ("tfidf", "binary") for mf in (100, 300, 1000, 3000) for p in grid]
    runs = Parallel(n_jobs=-1)(delayed(job)(k, mf, p, X_train, X_test, y_train, y_test) for k, mf, p in jobs)
    best = min(runs, key=lambda r: abs(r["accuracy"] - TARGET))
    near = [r["macro_f1"] for r in runs if abs(r["accuracy"] - TARGET) <= 0.01]
    print(f"best acc={best['accuracy']:.4f} macroF1={best['macro_f1']:.4f} {best['features']} "
          f"max_features={best['max_features']} {best['params']}")
    print(f"max acc over all runs={max(r['accuracy'] for r in runs):.4f}; runs within ±1pt: {len(near)}"
          + (f", macroF1 {min(near):.3f}-{max(near):.3f}" if near else ""))
    save_results("e05b_english_dt_small_vocab", {"runs": runs, "closest_run": best,
                                                 "macro_f1_range_within_1pt": [min(near), max(near)] if near else None},
                 paper={"accuracy": TARGET})


if __name__ == "__main__":
    main()
