"""Reproduce paper §6.2.2 / Table 3 — English aspect term extraction (SVM BIO tagger).

Original chain:
  Laptop_Train_v2.xml -> Laptop_Train_v2.jsonl   (format of extras/test2.py; file not in repo)
    -> english/prepare_dataset_en.py             -> english_ate_features.json
    -> english/training_with_grid_search_en.py   (GridSearch LinearSVC, 80/20 token split)

Paper: P 0.8347 / R 0.7803 / F1 0.8066, 97% token accuracy;
       Table 3: B-ASP 0.80/0.75/0.77, I-ASP 0.73/0.68/0.70, O 0.99/0.99/0.99.
Also reports which F1 definition reproduces 0.8066 (token-level vs exact span).
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

from _common import REPO, load_original, save_results

DATA_DIR = os.path.join(REPO, "experiments", "reviewer_revision", "data")
JSONL = os.path.join(DATA_DIR, "Laptop_Train_v2.jsonl")
FEATS = os.path.join(DATA_DIR, "english_ate_features.json")


def xml_to_jsonl():
    # extras/test2.py convert_xml_to_jsonl()
    with open(JSONL, "w", encoding="utf-8") as out:
        for s in ET.parse(os.path.join(REPO, "data", "raw", "semeval2014_laptops_train_v2.xml")).getroot().findall("sentence"):
            t = s.find("text")
            if t is None or not t.text:
                continue
            terms = [] if s.find("aspectTerms") is None else [
                {"term": a.get("term"), "polarity": a.get("polarity"),
                 "from": int(a.get("from")), "to": int(a.get("to"))}
                for a in s.find("aspectTerms").findall("aspectTerm")]
            out.write(json.dumps({"id": int(s.get("id")), "sentence": t.text.strip(),
                                  "aspect_terms": terms}, ensure_ascii=False) + "\n")


def spans(labels):
    """BIO sequence -> set of (start, end) token spans (I-ASP without B starts a span)."""
    out, start = set(), None
    for i, lab in enumerate(list(labels) + ["O"]):
        if lab == "B-ASP" or (lab == "I-ASP" and start is None):
            if start is not None:
                out.add((start, i))
            start = i
        elif lab == "O" and start is not None:
            out.add((start, i))
            start = None
    return out


def main():
    # optional argv[1]: alternative source jsonl (same id/sentence/aspect_terms format),
    # e.g. repro/data/cleaned_final_dataset.jsonl to test whether the paper used the Amazon data
    os.makedirs(DATA_DIR, exist_ok=True)
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if source is None:
        xml_to_jsonl()
    tag = "laptop" if source is None else os.path.splitext(os.path.basename(source))[0]
    feats_file = FEATS.replace(".json", f"_{tag}.json")

    prep = load_original("src/3_aspect_extraction/english/prepare_dataset_en.py", "prepare_dataset_en")
    prep.INPUT_FILE, prep.OUTPUT_FILE = source or JSONL, feats_file
    prep.prepare_dataset()

    tr = load_original("src/3_aspect_extraction/english/train_svm_gridsearch_en.py", "training_with_grid_search_en")
    tr.FEATURES_FILE = feats_file
    dataset = tr.load_dataset()
    feats, labels = tr.build_features(dataset)
    dist = {l: labels.count(l) for l in ("O", "B-ASP", "I-ASP")}
    print("tokens:", len(labels), dist, "sentences:", len(dataset))

    vec = DictVectorizer()
    X = vec.fit_transform(feats)
    y = np.array(labels)
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, idx, test_size=0.20, stratify=y, random_state=42)
    grid = GridSearchCV(LinearSVC(), {"C": [0.01, 0.1, 1, 2, 5], "class_weight": [None, "balanced"],
                                      "loss": ["hinge", "squared_hinge"], "max_iter": [2000, 3000, 5000]},
                        scoring="f1_macro", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True)
    print(grid.best_params_)
    print(classification_report(y_test, y_pred, digits=4))

    # candidate definitions of the paper's single P/R/F1
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y_test, y_pred, labels=["B-ASP", "I-ASP"], average="micro")
    yb_t = np.where(y_test == "O", 0, 1); yb_p = np.where(y_pred == "O", 0, 1)
    p_bin, r_bin, f_bin, _ = precision_recall_fscore_support(yb_t, yb_p, average="binary")
    candidates = {
        "token_micro_B+I": [p_mi, r_mi, f_mi],
        "token_binary_aspect_vs_O": [p_bin, r_bin, f_bin],
        "token_macro_all": [rep["macro avg"]["precision"], rep["macro avg"]["recall"], rep["macro avg"]["f1-score"]],
        "token_weighted_all": [rep["weighted avg"]["precision"], rep["weighted avg"]["recall"], rep["weighted avg"]["f1-score"]],
    }
    for k, v in candidates.items():
        print(f"{k:28s} P={v[0]:.4f} R={v[1]:.4f} F1={v[2]:.4f}")

    # Exact-span F1 is impossible on this token-level random split (sentences are cut across train/test),
    # so it is computed on a sentence-level split with the paper's best params (needed for the revision).
    sent_ids = np.concatenate([[k] * len(d["tokens"]) for k, d in enumerate(dataset)])
    s_train, s_test = train_test_split(np.arange(len(dataset)), test_size=0.20, random_state=42)
    m_tr = np.isin(sent_ids, s_train)
    vec2 = DictVectorizer()
    X2 = vec2.fit_transform([f for f, m in zip(feats, m_tr) if m])
    clf2 = LinearSVC(**grid.best_params_).fit(X2, y[m_tr])
    tp = fp = fn = 0
    tok_true, tok_pred = [], []
    pos = 0
    for k, d in enumerate(dataset):
        n = len(d["tokens"])
        if k in set(s_test.tolist()):
            pred = clf2.predict(vec2.transform(feats[pos:pos + n]))
            gold, hyp = spans(d["labels"]), spans(pred)
            tp += len(gold & hyp); fp += len(hyp - gold); fn += len(gold - hyp)
            tok_true += d["labels"]; tok_pred += list(pred)
        pos += n
    sp_p = tp / (tp + fp) if tp + fp else 0.0
    sp_r = tp / (tp + fn) if tp + fn else 0.0
    sp_f = 2 * sp_p * sp_r / (sp_p + sp_r) if sp_p + sp_r else 0.0
    print(f"sentence-split exact-span  P={sp_p:.4f} R={sp_r:.4f} F1={sp_f:.4f}")

    save_results(
        f"r05_english_aspect_extraction_{tag}",
        {"source": source or JSONL, "n_tokens": len(labels), "token_distribution": dist, "n_sentences": len(dataset),
         "best_params": grid.best_params_, "token_accuracy": accuracy_score(y_test, y_pred),
         "per_class": {k: rep[k] for k in ("B-ASP", "I-ASP", "O")},
         "single_prf_candidates": candidates,
         "sentence_split_exact_span": {"precision": sp_p, "recall": sp_r, "f1": sp_f,
                                       "tp": tp, "fp": fp, "fn": fn}},
        paper={"prf": [0.8347, 0.7803, 0.8066], "token_accuracy": 0.97,
               "B-ASP": [0.80, 0.75, 0.77], "I-ASP": [0.73, 0.68, 0.70], "O": [0.99, 0.99, 0.99],
               "fig3_confusion_binary": [[91709, 1587], [5103, 2606]]},
    )


if __name__ == "__main__":
    main()
