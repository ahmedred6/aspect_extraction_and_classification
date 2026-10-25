"""Is the SMOTE effect on Arabic aspect sentiment statistically significant?

Same setup as e01 (Stacking, 80/20 split, seed 42) with and without SMOTE; predictions kept so the two models
can be compared on the same test items with McNemar's exact test (overall, and within each class).
"""
import os

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from _common import REPO, load_original, save_results

ar = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")


def mcnemar(correct_a, correct_b):
    """Exact McNemar: only items where exactly one model is right count."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"with_smote_only_correct": a_only, "without_smote_only_correct": b_only, "p_value": p}


def main():
    df = ar.load_data(os.path.join(REPO, ar.DATA_PATH))
    df = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
    vec = ar.get_advanced_vectorizer()
    Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
    y = y_test.to_numpy()

    preds = {}
    for smote in (True, False):
        steps = ([("smote", SMOTE(random_state=42, k_neighbors=5))] if smote else []) + \
                [("stacking", ar.get_stacking_model())]
        preds[smote] = ImbPipeline(steps).fit(Xtr, y_train).predict(Xte)

    res = {}
    for smote, p in preds.items():
        res["with_smote" if smote else "without_smote"] = {
            "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "negative_recall": recall_score(y, p, pos_label="negative"),
            "positive_recall": recall_score(y, p, pos_label="positive")}
    ca, cb = preds[True] == y, preds[False] == y
    res["mcnemar_overall"] = mcnemar(ca, cb)
    for cls in ("negative", "positive"):
        m = y == cls
        res[f"mcnemar_{cls}_class"] = mcnemar(ca[m], cb[m])
    for k, v in res.items():
        print(k, v)
    save_results("e07_smote_significance", res)


if __name__ == "__main__":
    main()
