"""Reviewer item #6: is the English aspect-extraction F1 (0.8066) token-level or exact-span?

Data: SemEval-2014 Task 4 Laptop + Restaurant train (the closest match to Fig. 3's 101,005 tokens; see r06).
Preparation: regex tokenizer \\w+|[^\\w\\s], exact character offsets (closest variant in r06, 100,861 tokens).
Model: the team's tagger — english/features_en.py token features, DictVectorizer, LinearSVC tuned by the same
GridSearchCV as english/training_with_grid_search_en.py, on an 80/20 stratified TOKEN split (as in the script).

Reported for comparison with the paper (Table 3: B-ASP .80/.75/.77, I-ASP .73/.68/.70, O .99/.99/.99;
overall P .8347 / R .7803 / F1 .8066; 97% token accuracy):
  - per-tag token-level scores, and several token-level summaries of a single P/R/F1
  - exact-span P/R/F1 on a sentence-level 80/20 split (exact span cannot be scored on a token split)
"""
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

from _common import load_original, save_results
from r05_english_aspect_extraction import spans
from r06_english_ate_fingerprint import load

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
feat = load_original("src/3_aspect_extraction/english/features_en.py", "features_en").token_features
POLARITIES = {"pos/neg": ("positive", "negative"), "all": None}


def build(polarities):
    sents = []
    for item in load("laptop") + load("restaurant"):
        s = item["sentence"]
        toks, cmap = [], []
        for m in TOKEN_RE.finditer(s):
            toks.append(m.group()); cmap.append((m.start(), m.end()))
        labels = ["O"] * len(toks)
        for a in item["aspect_terms"]:
            if polarities and a["polarity"] not in polarities:
                continue
            inside = False
            for i, (cs, ce) in enumerate(cmap):
                if cs >= a["from"] and ce <= a["to"]:
                    labels[i] = "I-ASP" if inside else "B-ASP"
                    inside = True
        sents.append((toks, labels))
    return sents


def run(pol_name):
    sents = build(POLARITIES[pol_name])
    feats = [feat(t, i) for t, _ in sents for i in range(len(t))]
    y = np.array([l for _, ls in sents for l in ls])
    sid = np.concatenate([[k] * len(t) for k, (t, _) in enumerate(sents)])
    dist = {l: int((y == l).sum()) for l in ("O", "B-ASP", "I-ASP")}
    print(f"\n#### polarities={pol_name}: tokens={len(y)} {dist}", flush=True)

    # token-level protocol (the team's script)
    X = DictVectorizer().fit_transform(feats)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    grid = GridSearchCV(LinearSVC(), {"C": [0.01, 0.1, 1, 2, 5], "class_weight": [None, "balanced"],
                                      "loss": ["hinge", "squared_hinge"], "max_iter": [2000, 3000, 5000]},
                        scoring="f1_macro", cv=3, n_jobs=-1).fit(X_tr, y_tr)
    p = grid.best_estimator_.predict(X_te)
    rep = classification_report(y_te, p, output_dict=True)
    print(grid.best_params_)
    print(classification_report(y_te, p, digits=4))
    bt, bp = (y_te != "O").astype(int), (p != "O").astype(int)
    summaries = {
        "micro over B+I tags": precision_recall_fscore_support(y_te, p, labels=["B-ASP", "I-ASP"], average="micro")[:3],
        "macro over B+I tags": precision_recall_fscore_support(y_te, p, labels=["B-ASP", "I-ASP"], average="macro")[:3],
        "macro over B,I,O": precision_recall_fscore_support(y_te, p, average="macro")[:3],
        "aspect-vs-O binary": precision_recall_fscore_support(bt, bp, average="binary")[:3],
    }
    for k, v in summaries.items():
        print(f"  token-level {k:22s} P={v[0]:.4f} R={v[1]:.4f} F1={v[2]:.4f}")

    # exact-span protocol (sentence-level split, same best params)
    s_tr, s_te = train_test_split(np.arange(len(sents)), test_size=0.20, random_state=42)
    m_tr = np.isin(sid, s_tr)
    vec = DictVectorizer()
    clf = LinearSVC(**grid.best_params_).fit(vec.fit_transform([f for f, m in zip(feats, m_tr) if m]), y[m_tr])
    tp = fp = fn = 0
    for k in s_te:
        toks, gold = sents[k]
        pred = clf.predict(vec.transform([feat(toks, i) for i in range(len(toks))]))
        g, h = spans(gold), spans(pred)
        tp += len(g & h); fp += len(h - g); fn += len(g - h)
    sp_p, sp_r = tp / (tp + fp), tp / (tp + fn)
    span = {"precision": sp_p, "recall": sp_r, "f1": 2 * sp_p * sp_r / (sp_p + sp_r), "tp": tp, "fp": fp, "fn": fn}
    print(f"  exact-span (sentence split)          P={span['precision']:.4f} R={span['recall']:.4f} F1={span['f1']:.4f}")

    return {"n_tokens": len(y), "token_distribution": dist, "best_params": grid.best_params_,
            "token_accuracy": accuracy_score(y_te, p),
            "per_tag": {k: rep[k] for k in ("B-ASP", "I-ASP", "O")},
            "token_level_summaries": {k: list(map(float, v)) for k, v in summaries.items()},
            "exact_span": span}


def main():
    pols = sys.argv[1:] or list(POLARITIES)
    save_results("r07_english_ate_token_vs_span", {p: run(p) for p in pols},
                 paper={"overall": [0.8347, 0.7803, 0.8066], "token_accuracy": 0.97,
                        "B-ASP": [0.80, 0.75, 0.77], "I-ASP": [0.73, 0.68, 0.70], "O": [0.99, 0.99, 0.99]})


if __name__ == "__main__":
    main()
