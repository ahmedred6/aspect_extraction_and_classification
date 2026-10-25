"""Reviewer items: aspect sentiment — (1) results with vs without SMOTE, (2) macro-F1 alongside accuracy.

Same data, split (80/20 stratified, seed 42) and vectorizer as the reproduced runs r01 (Arabic) and
r03 (English). Every model is trained twice: with SMOTE (paper setting) and without SMOTE.
Models: Decision Tree, Linear SVM, Stacking (paper Table 8).

The Decision Tree / Linear SVM configurations behind Table 8 are not in the repo, so a few candidate
configurations are run and the one whose accuracy matches Table 8 is identified in the output.
"""
import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from _common import REPO, load_original, save_results

ar = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")
en = load_original("src/4_aspect_sentiment/aspect_sentiment_english.py", "AspClassEnglish")
RS = 42

PAPER_TABLE8 = {"english": {"Decision Tree": 0.765, "Linear SVM": 0.782, "Stacking": 0.800},
                "arabic": {"Decision Tree": 0.712, "Linear SVM": 0.855, "Stacking": 0.902}}


def candidates(mod):
    return {
        "Decision Tree": {
            "default": lambda: DecisionTreeClassifier(random_state=RS),
            "balanced": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=RS),
            "max_depth=20,balanced": lambda: DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=RS),
        },
        "Linear SVM": {
            "default": lambda: LinearSVC(random_state=RS),
            "balanced": lambda: LinearSVC(class_weight="balanced", random_state=RS),
            "C=0.5,balanced (stacking base)": lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=RS),
        },
        "Stacking": {"paper": mod.get_stacking_model},
    }


def evaluate(y_test, y_pred):
    rep = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "negative": {k: rep["negative"][k] for k in ("precision", "recall", "f1-score")},
        "positive": {k: rep["positive"][k] for k in ("precision", "recall", "f1-score")},
        "confusion_matrix[neg,pos]": confusion_matrix(y_test, y_pred, labels=["negative", "positive"]).tolist(),
    }


def run_language(lang, mod, df):
    df = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=mod.TEST_SIZE, stratify=df["label"], random_state=mod.RANDOM_STATE)
    vec = mod.get_advanced_vectorizer()
    Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
    print(f"\n######## {lang}: train={len(y_train)} test={len(y_test)} {y_train.value_counts().to_dict()}")

    out = {}
    for model_name, configs in candidates(mod).items():
        out[model_name] = {}
        for cfg_name, make in configs.items():
            out[model_name][cfg_name] = {}
            for smote in (True, False):
                steps = ([("smote", SMOTE(random_state=RS, k_neighbors=5))] if smote else []) + [("clf", make())]
                y_pred = ImbPipeline(steps).fit(Xtr, y_train).predict(Xte)
                r = evaluate(y_test, y_pred)
                out[model_name][cfg_name]["with_smote" if smote else "without_smote"] = r
                print(f"{model_name:14s} {cfg_name:32s} SMOTE={'yes' if smote else 'no ':3s} "
                      f"acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} negR={r['negative']['recall']:.4f}")
    # which candidate reproduces Table 8 (with SMOTE, the paper's setting)
    match = {m: min(cfgs, key=lambda c: abs(cfgs[c]["with_smote"]["accuracy"] - PAPER_TABLE8[lang][m]))
             for m, cfgs in out.items()}
    print("closest to Table 8:", {m: (c, round(out[m][c]["with_smote"]["accuracy"], 4)) for m, c in match.items()})
    return {"n_train": len(y_train), "n_test": len(y_test), "results": out, "closest_to_table8": match}


def main():
    results = {
        "arabic": run_language("arabic", ar, ar.load_data(os.path.join(REPO, ar.DATA_PATH))),
        "english": run_language("english", en, en.load_data(os.path.join(REPO, "repro", "data", "cleaned_final_dataset.jsonl"))),
    }
    save_results("e01_aspect_sentiment_smote_macro_f1", results,
                 paper={"table8_accuracy": PAPER_TABLE8, "arabic_negative_recall_with_smote": 0.8623})


if __name__ == "__main__":
    main()
