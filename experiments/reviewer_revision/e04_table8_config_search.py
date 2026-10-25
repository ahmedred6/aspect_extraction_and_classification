"""Find the Decision Tree / Linear SVM setup behind paper Table 8 (aspect sentiment accuracies).

No script in the repo produces those two rows, so every setup implied by the repo's aspect-sentiment
code is tried:
  features : "final"  = AspClass*.get_advanced_vectorizer (word 1-3 + char_wb 3-5), ASP_START/ASP_END markers
             "absa"   = absa/features.py build_vectorizer (word 1-2 + char 3-5), <ASP> markers (absa/, absa2/)
  models   : DT  {max_depth=15 (absa/models.py), max_depth=20+balanced (FINALWINNERYAY), default, balanced}
             SVM {C=1 (absa/models.py), balanced (aspClassEnglish2.py), C=0.5+balanced (stacking base)}
  SMOTE    : yes / no
  protocol : "test" = 80/20 stratified split (AspClass*.py)
             "cv5"  = 5-fold stratified CV on the training split, vectorizer fit on X_train (aspClassEnglish2.py)
A setup counts as a match only if it reproduces Table 8 for BOTH languages (within 0.3 points).
"""
import os

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from _common import REPO, load_original, save_results

ar = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")
en = load_original("src/4_aspect_sentiment/aspect_sentiment_english.py", "AspClassEnglish")
absa_feat = load_original("archive/early_aspect_sentiment_v1/features.py", "absa_features")
RS = 42
PAPER = {"english": {"DT": 0.765, "SVM": 0.782}, "arabic": {"DT": 0.712, "SVM": 0.855}}
TOL = 0.003

MODELS = {
    "DT": {"max_depth=15": lambda: DecisionTreeClassifier(max_depth=15, random_state=RS),
           "max_depth=20,balanced": lambda: DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=RS),
           "default": lambda: DecisionTreeClassifier(random_state=RS),
           "balanced": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=RS)},
    "SVM": {"C=1": lambda: LinearSVC(C=1.0, random_state=RS),
            "balanced": lambda: LinearSVC(class_weight="balanced"),
            "C=0.5,balanced": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RS)},
}


def data(lang, markers):
    mod = ar if lang == "arabic" else en
    df = (ar.load_data(os.path.join(REPO, ar.DATA_PATH)) if lang == "arabic"
          else en.load_data(os.path.join(REPO, "repro", "data", "cleaned_final_dataset.jsonl")))
    df = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    if markers == "absa":  # absa/aspect_windows.py uses <ASP> ... </ASP> (lowercased by clean_text)
        df["text"] = df["text"].str.replace("asp_start", "<asp>", regex=False).str.replace("asp_end", "</asp>", regex=False)
    return train_test_split(df["text"], df["label"], test_size=mod.TEST_SIZE, stratify=df["label"], random_state=RS)


def vectorizer(lang, features):
    return (ar if lang == "arabic" else en).get_advanced_vectorizer() if features == "final" else absa_feat.build_vectorizer()


def fit_predict(make, smote, Xtr, ytr, Xte):
    steps = ([("smote", SMOTE(random_state=RS))] if smote else []) + [("clf", make())]
    return ImbPipeline(steps).fit(Xtr, ytr).predict(Xte)


def main():
    rows = []
    for lang in ("english", "arabic"):
        for features in ("final", "absa"):
            X_train, X_test, y_train, y_test = data(lang, features)
            vec = vectorizer(lang, features)
            Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(Xtr, y_train))
            for family, cfgs in MODELS.items():
                for cfg, make in cfgs.items():
                    for smote in (True, False):
                        p = fit_predict(make, smote, Xtr, y_train, Xte)
                        test_acc, test_f1 = accuracy_score(y_test, p), f1_score(y_test, p, average="macro")
                        accs, f1s = [], []
                        for tr, va in folds:
                            pv = fit_predict(make, smote, Xtr[tr], y_train.iloc[tr], Xtr[va])
                            accs.append(accuracy_score(y_train.iloc[va], pv))
                            f1s.append(f1_score(y_train.iloc[va], pv, average="macro"))
                        for protocol, acc, f1 in (("test", test_acc, test_f1), ("cv5", np.mean(accs), np.mean(f1s))):
                            rows.append({"lang": lang, "family": family, "features": features, "model": cfg,
                                         "smote": smote, "protocol": protocol, "accuracy": float(acc), "macro_f1": float(f1)})
                        print(f"{lang:7s} {family:3s} {features:5s} {cfg:22s} smote={smote!s:5s} "
                              f"test={test_acc:.4f} cv5={np.mean(accs):.4f}", flush=True)

    # a setup matches if it hits Table 8 for both languages
    matches = {}
    for family in ("DT", "SVM"):
        keyed = {}
        for r in rows:
            if r["family"] == family:
                keyed.setdefault((r["features"], r["model"], r["smote"], r["protocol"]), {})[r["lang"]] = r
        matches[family] = [
            {"features": k[0], "model": k[1], "smote": k[2], "protocol": k[3],
             "english": v["english"], "arabic": v["arabic"]}
            for k, v in keyed.items()
            if all(abs(v[l]["accuracy"] - PAPER[l][family]) <= TOL for l in ("english", "arabic"))]
        print(f"\n{family} setups matching Table 8 in BOTH languages: {len(matches[family])}")
        for m in matches[family]:
            print("  ", m["features"], m["model"], "smote" if m["smote"] else "no-smote", m["protocol"],
                  f"EN acc={m['english']['accuracy']:.4f} F1={m['english']['macro_f1']:.4f} | "
                  f"AR acc={m['arabic']['accuracy']:.4f} F1={m['arabic']['macro_f1']:.4f}")
        if not matches[family]:
            for l in ("english", "arabic"):
                best = min((r for r in rows if r["family"] == family and r["lang"] == l),
                           key=lambda r: abs(r["accuracy"] - PAPER[l][family]))
                print(f"   closest {l}: {best}")

    save_results("e04_table8_config_search", {"runs": rows, "matches_both_languages": matches},
                 paper={"table8": PAPER, "tolerance": TOL})


if __name__ == "__main__":
    main()
