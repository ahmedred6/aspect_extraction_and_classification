"""How much does the token-level train/test split inflate aspect-extraction scores?

The paper's aspect-extraction results (Arabic Tables 5-6, English Table 3) split TOKENS 80/20, so tokens of the
same sentence (and their neighbours, which are features) appear in both train and test. Here the same models
are evaluated with a SENTENCE-level split (whole sentences go to train or test), and the vectorizer is fit on
training data only. Same data and model settings as the reproductions r04 (Arabic) and r07 (English).
"""
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from _common import load_original, save_results
from r04_arabic_aspect_extraction import FEATS as AR_FEATS
from r07_english_ate_token_vs_span import build as build_english
from r07_english_ate_token_vs_span import feat as en_feat


def metrics(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "B-ASP_f1": rep["B-ASP"]["f1-score"], "I-ASP_f1": rep["I-ASP"]["f1-score"],
            "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"]}


def evaluate(feats, y, groups, make_clf, name):
    y = np.asarray(y)
    out = {}
    # token-level split, vectorizer on all data (as in the paper's scripts)
    X = DictVectorizer().fit_transform(feats)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    out["token_split_holdout"] = metrics(y[te], make_clf().fit(X[tr], y[tr]).predict(X[te]))
    folds = [metrics(y[b], make_clf().fit(X[a], y[a]).predict(X[b]))
             for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)]
    out["token_split_cv5_mean"] = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}

    # sentence-level split, vectorizer fit on training sentences only
    def sentence_run(a, b):
        vec = DictVectorizer()
        Xa = vec.fit_transform([feats[i] for i in a])
        return metrics(y[b], make_clf().fit(Xa, y[a]).predict(vec.transform([feats[i] for i in b])))
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(feats, y, groups))
    out["sentence_split_holdout"] = sentence_run(a, b)
    folds = [sentence_run(a, b) for a, b in GroupKFold(5).split(feats, y, groups)]
    out["sentence_split_cv5_mean"] = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}

    print(f"\n#### {name}")
    for k, v in out.items():
        print(f"  {k:24s} macroF1={v['macro_f1']:.4f}  acc={v['accuracy']:.4f}  "
              f"B-ASP F1={v['B-ASP_f1']:.4f}  I-ASP F1={v['I-ASP_f1']:.4f}")
    return out


def main():
    # Arabic: Hamzah's loader groups tokens by sentence string (duplicate sentences merged), as in the paper
    ev = load_original("src/3_aspect_extraction/arabic/evaluate_svm_split.py", "evaluate_svm_split")
    ev.FEATURES_FILE = AR_FEATS
    dataset = ev.load_dataset()
    ar_feats, ar_y = ev.build_features(dataset)
    ar_groups = np.concatenate([[k] * len(d["tokens"]) for k, d in enumerate(dataset)])
    arabic = evaluate(ar_feats, ar_y, ar_groups, lambda: LinearSVC(class_weight="balanced"), "Arabic (paper: Tables 5-6)")

    # English: SemEval-2014 laptop+restaurant, all polarities, r07's best grid params
    sents = build_english(None)
    en_feats = [en_feat(t, i) for t, _ in sents for i in range(len(t))]
    en_y = [l for _, ls in sents for l in ls]
    en_groups = np.concatenate([[k] * len(t) for k, (t, _) in enumerate(sents)])
    english = evaluate(en_feats, en_y, en_groups,
                       lambda: LinearSVC(C=1, loss="hinge", max_iter=3000), "English (paper: Table 3)")

    save_results("e08_ate_split_leakage", {"arabic": arabic, "english": english},
                 paper={"arabic_holdout_macro_f1": 0.7163, "arabic_cv5_macro_f1": 0.7104, "arabic_accuracy": 0.9302,
                        "english_overall_f1": 0.8066, "english_B-ASP_f1": 0.77, "english_I-ASP_f1": 0.70})


if __name__ == "__main__":
    main()
