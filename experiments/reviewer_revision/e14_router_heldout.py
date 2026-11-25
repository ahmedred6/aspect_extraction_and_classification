"""Hamzah's review point #1: a real held-out test for the Smart Router, and a topic-vs-dialect check
(protocol D29, fixed before running).

(1) Held-out test: same filtering and class balancing as src/1_router/train_router.py, stratified 80/20 split
    (seed 42) BEFORE any fitting; the paper's winning configuration (char_wb TF-IDF 3-5, 10k + Logistic
    Regression, balanced) in one Pipeline, untuned; 5-fold CV on the training split only; test scored once.
(2) Topic check: the router data mixes LABR (book reviews) and HARD (hotel reviews). Each review's domain is
    labelled zero-shot by Qwen (BOOK / HOTEL / OTHER). Reported: label distribution per domain, a domain-only
    baseline (majority label per domain, fit on train), and router accuracy within each domain.
Needs QWEN_API_KEY in .env (domain labels are cached in results/e14_cache/).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import e11_llm_dialect_normalization as llm  # reuse the cached, retrying Qwen client
from _common import REPO, RESULTS_DIR, save_results

RS = 42
BATCH = 20
CACHE = os.path.join(RESULTS_DIR, "e14_cache", "domain_labels.jsonl")
DOMAIN_PROMPT = """For each numbered Arabic review, decide what is being reviewed.
Answer BOOK if it reviews a book, novel or author; HOTEL if it reviews a hotel, resort, room or stay;
otherwise OTHER. Return ONLY a JSON list, one object per review, in the same order:
[{"id": <number>, "domain": "BOOK" | "HOTEL" | "OTHER"}, ...]"""


def load_balanced():
    """Same steps as train_router.load_data: clean labels, keep MSA/DIALECT, downsample the majority (seed 42)."""
    df = pd.read_csv(os.path.join(REPO, "data", "raw", "router_dataset.csv"))
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["label", "text"])
    df = df[df["label"].isin(["MSA", "DIALECT"])]
    msa, dia = df[df["label"] == "MSA"], df[df["label"] == "DIALECT"]
    n = min(len(msa), len(dia))
    msa, dia = msa.sample(n=n, random_state=RS), dia.sample(n=n, random_state=RS)
    return pd.concat([msa, dia]).sample(frac=1, random_state=RS)  # keeps original row ids as index


def router():
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])


def domain_labels(df, key):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = {}
    if os.path.exists(CACHE):
        cache = {r["id"]: r["domain"] for r in map(json.loads, open(CACHE, encoding="utf-8"))}
    todo = [i for i in df.index if int(i) not in cache]
    print(f"[domain labels] cached {len(df) - len(todo)}, to do {len(todo)}", flush=True)
    for b in range(0, len(todo), BATCH):
        ids = todo[b:b + BATCH]
        listing = "\n".join(f"{k + 1}. {str(df.at[i, 'text'])[:800]}" for k, i in enumerate(ids))
        try:
            reply, _ = llm.chat([{"role": "system", "content": DOMAIN_PROMPT},
                                 {"role": "user", "content": listing}], 40 * len(ids) + 50, key)
            parsed = {int(o["id"]): str(o["domain"]).upper()
                      for o in json.loads(re.search(r"\[.*\]", reply, flags=re.S).group(0))}
        except (llm.Rejected, AttributeError, ValueError, KeyError, TypeError):
            parsed = {}
        with open(CACHE, "a", encoding="utf-8") as f:
            for k, i in enumerate(ids):
                d = parsed.get(k + 1)
                if d in ("BOOK", "HOTEL", "OTHER"):
                    f.write(json.dumps({"id": int(i), "domain": d}) + "\n")
        if (b // BATCH) % 25 == 0:
            print(f"  labelled {min(b + BATCH, len(todo))}/{len(todo)}", flush=True)
    cache = {r["id"]: r["domain"] for r in map(json.loads, open(CACHE, encoding="utf-8"))}
    return pd.Series({i: cache.get(int(i), "UNLABELLED") for i in df.index})


def main():
    df = load_balanced()
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.2,
                                              stratify=df["label"], random_state=RS)
    print(f"balanced {len(df)}: train {len(y_tr)}, test {len(y_te)}", flush=True)

    # (1) held-out test
    cv = cross_val_score(router(), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                         scoring="accuracy")
    model = router().fit(X_tr, y_tr)
    pred = pd.Series(model.predict(X_te), index=X_te.index)
    rep = classification_report(y_te, pred, output_dict=True)
    labels = ["DIALECT", "MSA"]
    heldout = {"train_cv5_accuracy_mean": float(cv.mean()), "train_cv5_accuracy_std": float(cv.std()),
               "test_accuracy": float(accuracy_score(y_te, pred)),
               "test_macro_f1": float(f1_score(y_te, pred, average="macro")),
               "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in labels},
               "confusion_matrix_rows_true_DIALECT_MSA": confusion_matrix(y_te, pred, labels=labels).tolist()}
    print(f"(1) train CV {cv.mean():.4f} ± {cv.std():.4f} | TEST acc {heldout['test_accuracy']:.4f} "
          f"macroF1 {heldout['test_macro_f1']:.4f} | CM {heldout['confusion_matrix_rows_true_DIALECT_MSA']}",
          flush=True)

    # (2) topic check
    domain = domain_labels(df, llm.api_key())
    dist = pd.crosstab(domain, df["label"]).to_dict()
    majority = (pd.DataFrame({"d": domain.loc[y_tr.index], "y": y_tr})
                .groupby("d")["y"].agg(lambda s: s.value_counts().idxmax()))
    base_pred = domain.loc[y_te.index].map(majority).fillna(y_tr.value_counts().idxmax())
    within = {}
    for d in sorted(domain.loc[y_te.index].unique()):
        m = domain.loc[y_te.index] == d
        within[d] = {"n_test": int(m.sum()), "label_counts": y_te[m].value_counts().to_dict(),
                     "router_accuracy": float(accuracy_score(y_te[m], pred[m])),
                     "router_macro_f1": float(f1_score(y_te[m], pred[m], average="macro"))
                     if y_te[m].nunique() > 1 else None}
    topic = {"domain_counts": domain.value_counts().to_dict(), "label_by_domain": dist,
             "domain_majority_label": majority.to_dict(),
             "domain_only_baseline_test_accuracy": float(accuracy_score(y_te, base_pred)),
             "router_within_domain": within}
    print(f"(2) domains {topic['domain_counts']}\n    labels by domain {dist}\n"
          f"    domain-only baseline test acc {topic['domain_only_baseline_test_accuracy']:.4f}\n"
          f"    router within domain {within}", flush=True)

    save_results("e14_router_heldout", {"heldout": heldout, "topic_check": topic},
                 paper={"cv_accuracy": 0.8392, "training_fit": 0.8858})


if __name__ == "__main__":
    main()
