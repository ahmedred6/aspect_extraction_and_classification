"""Hamzah's review point #4: Table 8 baselines under one protocol fixed before running (decision D19).

e05 rebuilt the Decision Tree / Linear SVM rows by sweeping settings and reporting the run closest to the paper's
accuracy. That is selection toward a target. Here the protocol is fixed in advance and run once:

  1. Data and split: identical to the Stacking row (neutral/conflict removed; 80/20 stratified, seed 42).
  2. Features: identical to Stacking for BOTH languages: get_advanced_vectorizer()
     (word TF-IDF 1-3-grams, 20k + char_wb TF-IDF 3-5-grams, 30k), fit on training data only.
  3. SMOTE (k=5, seed 42) on training data for all three models, as for Stacking.
  4. Tuning: DT and SVM choose from a small grid by 5-fold stratified CV on the TRAINING split only
     (DT: max_depth {None, 20, 50} x min_samples_leaf {1, 5}; SVM: C {0.1, 0.5, 1, 2}).
     Stacking keeps the paper's fixed configuration (not tuned), so the comparison is conservative.
  5. The test set is scored once: accuracy and macro-F1; McNemar exact tests DT vs Stacking and SVM vs Stacking.
     Plus 5-fold stratified CV over all data (vectorizer + SMOTE + model refit in every fold): mean and std.

Whatever comes out is reported.
"""
import os

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from _common import REPO, load_original, save_results

RS = 42
DATA = {"arabic": None, "english": os.path.join(REPO, "experiments", "reviewer_revision", "data",
                                                 "cleaned_final_dataset.jsonl")}
MODULES = {"arabic": "src/4_aspect_sentiment/aspect_sentiment_arabic.py",
           "english": "src/4_aspect_sentiment/aspect_sentiment_english.py"}
GRIDS = {"Decision Tree": {"clf__max_depth": [None, 20, 50], "clf__min_samples_leaf": [1, 5]},
         "Linear SVM": {"clf__C": [0.1, 0.5, 1, 2]}}
PAPER = {"english": {"Decision Tree": 0.765, "Linear SVM": 0.782, "Stacking Ensemble": 0.800},
         "arabic": {"Decision Tree": 0.712, "Linear SVM": 0.855, "Stacking Ensemble": 0.902}}


def pipeline(mod, family, **params):
    clf = {"Decision Tree": lambda: DecisionTreeClassifier(random_state=RS, **params),
           "Linear SVM": lambda: LinearSVC(random_state=RS, max_iter=5000, **params),
           "Stacking Ensemble": lambda: mod.get_stacking_model()}[family]()
    return ImbPipeline([("vec", mod.get_advanced_vectorizer()),
                        ("smote", SMOTE(random_state=RS, k_neighbors=5)),
                        ("clf", clf)])


def mcnemar(correct_a, correct_b):
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"only_baseline_correct": a_only, "only_stacking_correct": b_only, "p_value": float(p)}


def run(lang):
    mod = load_original(MODULES[lang], f"aspect_sentiment_{lang}")
    df = mod.load_data(DATA[lang] or os.path.join(REPO, mod.DATA_PATH))
    df = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    X, y = df["text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    print(f"\n######## {lang}: {len(df)} samples (train {len(y_tr)}, test {len(y_te)})", flush=True)

    inner = StratifiedKFold(5, shuffle=True, random_state=RS)
    chosen = {}
    for family, grid in GRIDS.items():
        gs = GridSearchCV(pipeline(mod, family), grid, cv=inner, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
        chosen[family] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        print(f"  {family}: chosen on training CV {chosen[family]} (inner macro-F1 {gs.best_score_:.4f})", flush=True)
    chosen["Stacking Ensemble"] = {}

    out, correct = {"n_train": len(y_tr), "n_test": len(y_te), "rows": {}}, {}
    y_te_arr = y_te.to_numpy()
    for family, params in chosen.items():
        p = pipeline(mod, family, **params).fit(X_tr, y_tr).predict(X_te)
        correct[family] = p == y_te_arr
        out["rows"][family] = {"params": params, "test_accuracy": float(accuracy_score(y_te, p)),
                               "test_macro_f1": float(f1_score(y_te, p, average="macro")),
                               "paper_accuracy": PAPER[lang][family]}
        cv = cross_validate(pipeline(mod, family, **params), X, y, scoring=["accuracy", "f1_macro"],
                            cv=StratifiedKFold(5, shuffle=True, random_state=RS), n_jobs=-1)
        out["rows"][family]["cv5"] = {
            "accuracy_mean": float(cv["test_accuracy"].mean()), "accuracy_std": float(cv["test_accuracy"].std()),
            "macro_f1_mean": float(cv["test_f1_macro"].mean()), "macro_f1_std": float(cv["test_f1_macro"].std())}
        r = out["rows"][family]
        print(f"  {family:18s} test acc={r['test_accuracy']:.4f} macroF1={r['test_macro_f1']:.4f} | "
              f"cv5 acc={r['cv5']['accuracy_mean']:.4f}±{r['cv5']['accuracy_std']:.4f} "
              f"macroF1={r['cv5']['macro_f1_mean']:.4f}±{r['cv5']['macro_f1_std']:.4f} | paper {PAPER[lang][family]}",
              flush=True)

    out["mcnemar_vs_stacking"] = {f: mcnemar(correct[f], correct["Stacking Ensemble"])
                                  for f in ("Decision Tree", "Linear SVM")}
    for f, m in out["mcnemar_vs_stacking"].items():
        print(f"  McNemar {f} vs Stacking: {m}")
    return out


def main():
    save_results("e10_table8_fair_baselines", {lang: run(lang) for lang in ("english", "arabic")}, paper=PAPER)


if __name__ == "__main__":
    main()
