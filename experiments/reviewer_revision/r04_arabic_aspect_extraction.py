"""Reproduce paper §6.5.3-§6.6 / Tables 5-7 — Arabic aspect term extraction (SVM BIO tagger).

Original chain:
  Hamzah`s_Part/Arabic_Hotels_TrD_V2.xml
    -> Arabic_Hotels_TrD_V2.jsonl          (conversion script NOT in repo; rebuilt here)
    -> extras/biotagger.py                 (fixed BIO tagging) -> arabic_ate_features_fixed.json
    -> Hamzah`s_Part/evaluate_svm_split.py (80/20 token-level split)  -> Table 5
    -> Hamzah`s_Part/train_svm_ate.py      (5-fold token-level CV)    -> Table 6
    -> extras/ablation_normalization.py    (with / without normalize) -> Table 7

Paper: tokens 106,876 = 97,235 O / 8,139 B-ASP / 1,502 I-ASP;
       held-out acc 0.9302, macro-F1 0.7163; 5-fold mean macro-F1 0.7104, acc 0.9302;
       ablation macro-F1 0.7185 (with) vs 0.7132 (without).
"""
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from _common import REPO, load_original, save_results

DATA_DIR = os.path.join(REPO, "experiments", "reviewer_revision", "data")
XML = os.path.join(REPO, "data", "raw", "semeval2016_arabic_hotels_train_v2.xml")
JSONL = os.path.join(DATA_DIR, "Arabic_Hotels_TrD_V2.jsonl")
FEATS = os.path.join(DATA_DIR, "arabic_ate_features_fixed.json")

PAPER_DIST = {"O": 97235, "B-ASP": 8139, "I-ASP": 1502}


def xml_to_jsonl(include_no_opinions):
    rows = []
    for sent in ET.parse(XML).getroot().iter("sentence"):
        text = sent.findtext("text")
        if not text:
            continue
        ops = sent.find("Opinions")
        aspects = [] if ops is None else [
            {"aspectTerm": o.get("target"), "polarity": o.get("polarity"),
             "category": o.get("category"), "from": o.get("from"), "to": o.get("to")}
            for o in ops.findall("Opinion")]
        if ops is None and not include_no_opinions:
            continue
        rows.append({"id": sent.get("id"), "sentence": text, "aspects": aspects})
    return rows


def bio_tag(rows, normalize, bt):
    """extras/biotagger.py prepare_dataset() (normalize=True) / ablation build_dataset()."""
    samples = []
    for sid, item in enumerate(rows):
        sentence = bt.normalize_arabic(item["sentence"]) if normalize else item["sentence"]
        tokens, char_map = bt.build_char_map(sentence)
        labels = ["O"] * len(tokens)
        for asp in item["aspects"]:
            if asp["polarity"] not in ["positive", "negative"]:
                continue
            term = bt.normalize_arabic(asp["aspectTerm"]) if normalize else asp["aspectTerm"]
            for (start, end) in bt.find_all_occurrences(sentence, term):
                inside = False
                for i, (cs, ce) in enumerate(char_map):
                    if cs >= start and ce <= end:
                        labels[i] = "I-ASP" if inside else "B-ASP"
                        inside = True
        for tok, lab in zip(tokens, labels):
            samples.append({"token": tok, "label": lab, "sentence": sentence, "sid": sid})
    return samples


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # biotagger.py prints with tqdm at import-time only inside prepare_dataset, safe to import
    bt = load_original("src/3_aspect_extraction/arabic/prepare_bio_dataset.py", "biotagger")

    # 1. find the XML->JSONL variant whose token distribution matches the paper
    variants = {}
    for include in (True, False):
        rows = xml_to_jsonl(include)
        dist = Counter(s["label"] for s in bio_tag(rows, True, bt))
        variants[f"include_sentences_without_opinions={include}"] = {
            "sentences": len(rows), "tokens": sum(dist.values()), **dist}
        print(include, len(rows), sum(dist.values()), dict(dist))
    chosen = min(variants, key=lambda k: sum(abs(variants[k][l] - v) for l, v in PAPER_DIST.items()))
    include = chosen.endswith("True")
    rows = xml_to_jsonl(include)
    with open(JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    samples = bio_tag(rows, True, bt)
    with open(FEATS, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    dist = Counter(s["label"] for s in samples)
    print(f"chosen: {chosen} -> {dict(dist)} total={len(samples)}")

    # 2. Hamzah's own loaders on the rebuilt features file
    ev = load_original("src/3_aspect_extraction/arabic/evaluate_svm_split.py", "evaluate_svm_split")
    ev.FEATURES_FILE = FEATS
    feats, labels = ev.build_features(ev.load_dataset())
    vec = DictVectorizer()
    X = vec.fit_transform(feats)
    y = np.array(labels)

    # Table 5 — evaluate_svm_split.py
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf = LinearSVC(class_weight="balanced").fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    t5 = classification_report(y_te, y_pred, output_dict=True)
    held = {"accuracy": accuracy_score(y_te, y_pred), "macro_f1": f1_score(y_te, y_pred, average="macro"),
            "per_class": {k: t5[k] for k in ("B-ASP", "I-ASP", "O")}}
    print(classification_report(y_te, y_pred, digits=4))
    print("held-out:", held["accuracy"], held["macro_f1"])

    # Table 6 — train_svm_ate.py (5-fold)
    folds = []
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
        p = LinearSVC(class_weight="balanced").fit(X[tr], y[tr]).predict(X[te])
        folds.append({"macro_f1": f1_score(y[te], p, average="macro"), "accuracy": accuracy_score(y[te], p)})
        print(folds[-1])
    cv = {"folds": folds, "mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
          "mean_accuracy": float(np.mean([f["accuracy"] for f in folds]))}

    # Table 7 — extras/ablation_normalization.py (its own pipeline: no sentence grouping, C=1, max_iter=5000)
    ab = {}
    for norm in (True, False):
        s = bio_tag(rows, norm, bt)
        # ablation script computes features per sentence directly (no grouping by sentence string)
        by_sid = {}
        for row in s:
            by_sid.setdefault(row["sid"], []).append(row)
        f_, l_ = [], []
        for toks in by_sid.values():
            tokens = [t["token"] for t in toks]
            for i, t in enumerate(toks):
                f_.append(bt_token_features(tokens, i))
                l_.append(t["label"])
        Xa = DictVectorizer().fit_transform(f_)
        ya = np.array(l_)
        a_tr, a_te, b_tr, b_te = train_test_split(Xa, ya, test_size=0.20, random_state=42, stratify=ya)
        pa = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000).fit(a_tr, b_tr).predict(a_te)
        rep = classification_report(b_te, pa, output_dict=True)
        ab["with" if norm else "without"] = {"macro_f1": f1_score(b_te, pa, average="macro"),
                                             "B-ASP_f1": rep["B-ASP"]["f1-score"],
                                             "I-ASP_f1": rep["I-ASP"]["f1-score"]}
        print("ablation", norm, ab["with" if norm else "without"])

    save_results(
        "r04_arabic_aspect_extraction",
        {"xml_variants": variants, "chosen_variant": chosen, "token_distribution": dict(dist),
         "n_tokens": len(samples), "held_out": held, "cv5": cv, "ablation": ab},
        paper={"token_distribution": PAPER_DIST, "n_tokens": 106876,
               "held_out": {"accuracy": 0.9302, "macro_f1": 0.7163,
                            "B-ASP": [0.59, 0.79, 0.68, 1628], "I-ASP": [0.47, 0.55, 0.51, 300],
                            "O": [0.98, 0.95, 0.96, 19448]},
               "cv5": {"mean_macro_f1": 0.7104, "mean_accuracy": 0.9302,
                       "folds_macro_f1": [0.7127, 0.7129, 0.7139, 0.7082, 0.7041]},
               "ablation": {"with": {"macro_f1": 0.7185, "B-ASP_f1": 0.6762},
                            "without": {"macro_f1": 0.7132, "B-ASP_f1": 0.6794}}},
    )


def bt_token_features(tokens, idx):
    token = tokens[idx]
    feats = {"token": token, "lower": token.lower(), "isdigit": token.isdigit(),
             "prefix1": token[:1], "prefix2": token[:2], "suffix1": token[-1:], "suffix2": token[-2:]}
    feats["prev"] = tokens[idx - 1].lower() if idx > 0 else "<START>"
    feats["next"] = tokens[idx + 1].lower() if idx < len(tokens) - 1 else "<END>"
    return feats


if __name__ == "__main__":
    main()
