"""Hamzah's review point #2: aspect-extraction metrics that do not depend on the easy O class.

The paper reports token-level macro-F1 over B-ASP / I-ASP / O on a TOKEN-level 80/20 split (Arabic 0.7163,
English 0.8066). This adds, for both languages, without changing the models:
  - exact-span precision / recall / F1 (an aspect counts only if the whole phrase matches);
  - per-tag token-level P/R/F1 for B-ASP, I-ASP and O;
  - macro-F1 over B-ASP and I-ASP only (O excluded);
  - the paper's macro-F1 over B, I, O, for continuity.

Protocols:
  token_split_holdout     the paper's split (tokens shuffled 80/20, vectorizer on all data). Exact span cannot be
                          scored here because a sentence's tokens are spread over train and test.
  sentence_split_holdout  whole sentences 80/20 (GroupShuffleSplit, seed 42), vectorizer fit on train only.
  sentence_split_cv5      5-fold GroupKFold over sentences; mean and std.
Models:
  paper_settings          Arabic LinearSVC(class_weight="balanced") as in evaluate_svm_split.py;
                          English LinearSVC(C=1, loss="hinge", max_iter=3000), r07's grid-search best.
  grouped_tuned           small grid chosen by 3-fold GroupKFold on the TRAINING sentences only (no test data).
Sentence grouping: Arabic uses the team's loader (duplicate sentence strings are merged, as in the paper);
English groups identical sentence texts so duplicates never straddle train and test.

Also writes a preview of a regenerated Fig. 3 (English aspect-vs-non-aspect confusion matrix on the sentence-level
test split) to results/figures/. It is NOT inserted into the paper.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from _common import RESULTS_DIR, load_original, save_results
from r04_arabic_aspect_extraction import FEATS as AR_FEATS
from r05_english_aspect_extraction import spans
from r07_english_ate_token_vs_span import build as build_english
from r07_english_ate_token_vs_span import feat as en_feat

TAGS = ["B-ASP", "I-ASP", "O"]
GRID = {"C": [0.1, 1, 2], "class_weight": [None, "balanced"], "loss": ["hinge", "squared_hinge"]}


# ---------------------------------------------------------------- metrics
def token_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=TAGS, zero_division=0)
    per_tag = {t: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
               for i, t in enumerate(TAGS)}
    bi = precision_recall_fscore_support(y_true, y_pred, labels=["B-ASP", "I-ASP"], average="macro", zero_division=0)
    bio = precision_recall_fscore_support(y_true, y_pred, labels=TAGS, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "per_tag": per_tag,
            "macro_BI": {"precision": float(bi[0]), "recall": float(bi[1]), "f1": float(bi[2])},
            "macro_BIO": {"precision": float(bio[0]), "recall": float(bio[1]), "f1": float(bio[2])}}


def span_metrics(gold_seqs, pred_seqs):
    tp = fp = fn = 0
    for g, h in zip(gold_seqs, pred_seqs):
        g, h = spans(g), spans(h)
        tp += len(g & h); fp += len(h - g); fn += len(g - h)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------- sentence-level training / scoring
def flatten(sents, idx, featfn):
    X = [featfn(sents[k][0], i) for k in idx for i in range(len(sents[k][0]))]
    y = [lab for k in idx for lab in sents[k][1]]
    return X, np.array(y)


def fit_predict(sents, tr, te, featfn, params):
    vec = DictVectorizer()
    Xtr, ytr = flatten(sents, tr, featfn)
    clf = LinearSVC(**params).fit(vec.fit_transform(Xtr), ytr)
    preds = []
    for k in te:
        toks = sents[k][0]
        preds.append(clf.predict(vec.transform([featfn(toks, i) for i in range(len(toks))])))
    gold = [sents[k][1] for k in te]
    return gold, preds, clf


def score_sentences(gold, preds):
    out = token_metrics(np.concatenate([np.asarray(g) for g in gold]), np.concatenate(preds))
    out["exact_span"] = span_metrics(gold, preds)
    return out


def grouped_tune(sents, tr, groups, featfn, base):
    """Pick params by macro-F1 over B/I on 3-fold GroupKFold inside the training sentences only."""
    tr = np.asarray(tr)
    best, best_score = None, -1
    for params in ParameterGrid(GRID):
        params = {**base, **params}
        scores = []
        for a, b in GroupKFold(3).split(tr, groups=groups[tr]):
            gold, preds, _ = fit_predict(sents, tr[a], tr[b], featfn, params)
            scores.append(token_metrics(np.concatenate([np.asarray(g) for g in gold]),
                                        np.concatenate(preds))["macro_BI"]["f1"])
        if np.mean(scores) > best_score:
            best, best_score = params, float(np.mean(scores))
    return best, best_score


def mean_std(runs):
    """Mean and std of the headline numbers over CV folds."""
    keys = {"exact_span_f1": lambda m: m["exact_span"]["f1"], "macro_BI_f1": lambda m: m["macro_BI"]["f1"],
            "macro_BIO_f1": lambda m: m["macro_BIO"]["f1"], "B-ASP_f1": lambda m: m["per_tag"]["B-ASP"]["f1"],
            "I-ASP_f1": lambda m: m["per_tag"]["I-ASP"]["f1"]}
    return {k: {"mean": float(np.mean([f(r) for r in runs])), "std": float(np.std([f(r) for r in runs]))}
            for k, f in keys.items()}


# ---------------------------------------------------------------- one language
def evaluate(name, sents, groups, featfn, paper_params, base):
    print(f"\n######## {name}: {len(sents)} sentences, {sum(len(t) for t, _ in sents)} tokens", flush=True)
    out = {"n_sentences": len(sents), "n_tokens": int(sum(len(t) for t, _ in sents))}
    idx = np.arange(len(sents))

    # A. paper's token-level split (token metrics only)
    feats = [featfn(t, i) for t, _ in sents for i in range(len(t))]
    y = np.array([lab for _, ls in sents for lab in ls])
    X = DictVectorizer().fit_transform(feats)
    a, b = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    out["token_split_holdout"] = token_metrics(y[b], LinearSVC(**paper_params).fit(X[a], y[a]).predict(X[b]))

    # B. sentence-level hold-out
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(idx, groups=groups))
    gold, preds, _ = fit_predict(sents, tr, te, featfn, paper_params)
    out["sentence_split_holdout"] = {"paper_settings": score_sentences(gold, preds)}
    out["_holdout_preds"] = (gold, preds)

    tuned, inner = grouped_tune(sents, tr, groups, featfn, base)
    gold_t, preds_t, _ = fit_predict(sents, tr, te, featfn, tuned)
    out["sentence_split_holdout"]["grouped_tuned"] = {**score_sentences(gold_t, preds_t),
                                                      "params": tuned, "inner_cv_macro_BI_f1": inner}

    # C. sentence-level 5-fold CV
    folds = [score_sentences(*fit_predict(sents, a, b, featfn, paper_params)[:2])
             for a, b in GroupKFold(5).split(idx, groups=groups)]
    out["sentence_split_cv5"] = {"paper_settings": mean_std(folds)}

    for proto in ("token_split_holdout",):
        m = out[proto]
        print(f"  {proto:34s} macroBIO={m['macro_BIO']['f1']:.4f}  macroBI={m['macro_BI']['f1']:.4f}")
    for variant, m in out["sentence_split_holdout"].items():
        print(f"  sentence_holdout/{variant:17s} span F1={m['exact_span']['f1']:.4f}  "
              f"macroBI={m['macro_BI']['f1']:.4f}  macroBIO={m['macro_BIO']['f1']:.4f}  "
              f"B={m['per_tag']['B-ASP']['f1']:.4f}  I={m['per_tag']['I-ASP']['f1']:.4f}")
    cv = out["sentence_split_cv5"]["paper_settings"]
    print("  sentence_cv5/paper_settings       " +
          "  ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in cv.items()))
    return out


def fig3_preview(gold, preds, path):
    yt = np.concatenate([np.asarray(g) for g in gold]) != "O"
    yp = np.concatenate(preds) != "O"
    cm = confusion_matrix(yt.astype(int), yp.astype(int))
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["non-aspect", "aspect"]).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("English aspect extraction\n(sentence-level test split)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"matrix": cm.tolist(), "aspect_recall": float(cm[1, 1] / cm[1].sum()),
            "aspect_precision": float(cm[1, 1] / cm[:, 1].sum()), "n_tokens": int(cm.sum())}


def main():
    # Arabic: the team's loader (sentences grouped by string, as for Tables 5-6)
    ev = load_original("src/3_aspect_extraction/arabic/evaluate_svm_split.py", "evaluate_svm_split")
    ev.FEATURES_FILE = AR_FEATS
    ar_sents = [(d["tokens"], d["labels"]) for d in ev.load_dataset()]
    ar_groups = np.arange(len(ar_sents))  # already one entry per distinct sentence string
    ar_feat = load_original("src/3_aspect_extraction/arabic/features.py", "features").token_features
    arabic = evaluate("Arabic (SemEval-2016 hotels; paper Tables 5-6)", ar_sents, ar_groups, ar_feat,
                      {"class_weight": "balanced"}, {"max_iter": 5000})

    # English: SemEval-2014 laptop + restaurant, all polarities (r07); identical sentences share a group
    en_sents = build_english(None)
    texts = [" ".join(t) for t, _ in en_sents]
    lookup = {s: k for k, s in enumerate(dict.fromkeys(texts))}
    en_groups = np.array([lookup[s] for s in texts])
    english = evaluate("English (SemEval-2014 laptop+restaurant; paper Table 3)", en_sents, en_groups, en_feat,
                       {"C": 1, "loss": "hinge", "max_iter": 3000}, {"max_iter": 5000})

    fig = fig3_preview(*english.pop("_holdout_preds"), os.path.join(RESULTS_DIR, "figures", "e09_fig3_preview.png"))
    arabic.pop("_holdout_preds")
    print(f"\nFig. 3 preview: aspect recall {fig['aspect_recall']:.3f}, precision {fig['aspect_precision']:.3f}, "
          f"{fig['n_tokens']} test tokens")

    save_results("e09_ate_span_metrics", {"arabic": arabic, "english": english, "fig3_preview": fig},
                 paper={"arabic_holdout_macro_BIO_f1": 0.7163, "english_macro_BIO_f1": 0.8066,
                        "english_B-ASP_f1": 0.77, "english_I-ASP_f1": 0.70,
                        "fig3_original": {"matrix": [[91709, 1587], [5103, 2606]], "aspect_recall": 0.338}})


if __name__ == "__main__":
    main()
