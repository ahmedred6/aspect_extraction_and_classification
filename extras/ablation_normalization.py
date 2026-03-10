"""
ablation_normalization.py
=========================
Runs the Arabic Aspect Term Extraction SVM under two conditions:

  CONDITION A — WITH normalize_arabic() (what the paper describes)
  CONDITION B — WITHOUT normalization (raw Arabic text)

Both use the same LinearSVC + DictVectorizer + token_features setup
and the same 80/20 stratified split (random_state=42) so results
are directly comparable.

Outputs a side-by-side table suitable for pasting into the paper.

Run from the project root (or from Hamzah`s_Part/):
    python ablation_normalization.py
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE = "Arabic_Hotels_TrD_V2.jsonl"
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "Hamzah`s_Part/Arabic_Hotels_TrD_V2.jsonl"

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ── normalizer ────────────────────────────────────────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"ـ", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ي", "ي").replace("ك", "ك")
    text = re.sub(r"[^\w\s،؟.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text)

def build_char_map(text):
    tokens, char_map = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group())
        char_map.append((m.start(), m.end()))
    return tokens, char_map

def find_all_occurrences(text, sub):
    positions = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(sub)))
        start = idx + 1
    return positions


# ── token features (same as features.py) ──────────────────────────────────────
def token_features(tokens, idx):
    token = tokens[idx]
    feats = {
        "token":   token,
        "lower":   token.lower(),
        "isdigit": token.isdigit(),
        "prefix1": token[:1],
        "prefix2": token[:2],
        "suffix1": token[-1:],
        "suffix2": token[-2:],
    }
    feats["prev"] = tokens[idx - 1].lower() if idx > 0 else "<START>"
    feats["next"] = tokens[idx + 1].lower() if idx < len(tokens) - 1 else "<END>"
    return feats


# ── dataset builder ───────────────────────────────────────────────────────────
def build_dataset(normalize: bool):
    """
    Returns (features_list, labels_list).
    If normalize=True, applies normalize_arabic() to sentence and aspect terms
    before computing BIO tags (fixed char_map logic).
    If normalize=False, uses raw text throughout.
    """
    samples_feats  = []
    samples_labels = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        item = json.loads(line)
        raw_sentence = item["sentence"]

        # Apply or skip normalization
        sentence = normalize_arabic(raw_sentence) if normalize else raw_sentence

        tokens, char_map = build_char_map(sentence)
        labels = ["O"] * len(tokens)

        filtered_aspects = [
            asp for asp in item["aspects"]
            if asp["polarity"] in ["positive", "negative"]
        ]

        for asp in filtered_aspects:
            term_raw  = asp["aspectTerm"]
            term      = normalize_arabic(term_raw) if normalize else term_raw
            occurrences = find_all_occurrences(sentence, term)

            for (start, end) in occurrences:
                inside = False
                for i, (cstart, cend) in enumerate(char_map):
                    if cstart >= start and cend <= end:
                        if not inside:
                            labels[i] = "B-ASP"
                            inside = True
                        else:
                            labels[i] = "I-ASP"

        for i, (tok, lab) in enumerate(zip(tokens, labels)):
            samples_feats.append(token_features(tokens, i))
            samples_labels.append(lab)

    return samples_feats, samples_labels


# ── evaluation ────────────────────────────────────────────────────────────────
def run_condition(normalize: bool):
    label = "WITH normalization" if normalize else "WITHOUT normalization"
    print(f"\n{'='*60}")
    print(f"  CONDITION: {label}")
    print(f"{'='*60}")

    feats, labels = build_dataset(normalize=normalize)

    dist = Counter(labels)
    print(f"  Token distribution: {dict(dist)}")

    vec = DictVectorizer()
    X   = vec.fit_transform(feats)
    y   = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Extract key metrics
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    metrics = {
        "condition":    label,
        "macro_f1":     f1_score(y_test, y_pred, average="macro"),
        "accuracy":     accuracy_score(y_test, y_pred),
        "B-ASP_f1":     report.get("B-ASP", {}).get("f1-score", 0.0),
        "B-ASP_prec":   report.get("B-ASP", {}).get("precision", 0.0),
        "B-ASP_rec":    report.get("B-ASP", {}).get("recall", 0.0),
        "I-ASP_f1":     report.get("I-ASP", {}).get("f1-score", 0.0),
        "O_f1":         report.get("O",     {}).get("f1-score", 0.0),
        "basp_support": int(report.get("B-ASP", {}).get("support", 0)),
    }
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  NORMALIZATION ABLATION STUDY")
    print("  Arabic Aspect Term Extraction — LinearSVC + token features")
    print("  Data: SemEval-2016 Arabic Hotels (TrD V2)")
    print("  Split: 80/20 stratified, random_state=42")
    print("="*60)

    results_with    = run_condition(normalize=True)
    results_without = run_condition(normalize=False)

    # ── summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  ABLATION SUMMARY TABLE")
    print("="*60)
    print(f"  {'Metric':<22} {'With Norm':>12} {'Without Norm':>14} {'Delta':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*14} {'-'*10}")

    metrics_to_show = [
        ("Macro F1",    "macro_f1"),
        ("Accuracy",    "accuracy"),
        ("B-ASP F1",    "B-ASP_f1"),
        ("B-ASP Prec",  "B-ASP_prec"),
        ("B-ASP Rec",   "B-ASP_rec"),
        ("I-ASP F1",    "I-ASP_f1"),
        ("O F1",        "O_f1"),
    ]

    for display_name, key in metrics_to_show:
        w  = results_with[key]
        wo = results_without[key]
        delta = w - wo
        sign  = "+" if delta >= 0 else ""
        print(f"  {display_name:<22} {w:>12.4f} {wo:>14.4f} {sign}{delta:>9.4f}")

    print(f"\n  B-ASP support (test set, normalized): {results_with['basp_support']}")

    print("\n\n" + "="*60)
    print("  PAPER-READY SUMMARY")
    print("="*60)
    print(f"""
  Ablation on normalization (Table X):

  System                  | Macro F1 | B-ASP F1 | B-ASP P | B-ASP R
  ----------------------- | -------- | -------- | ------- | -------
  With normalize_arabic() | {results_with['macro_f1']:.4f}   | {results_with['B-ASP_f1']:.4f}   | {results_with['B-ASP_prec']:.4f}  | {results_with['B-ASP_rec']:.4f}
  Without normalization   | {results_without['macro_f1']:.4f}   | {results_without['B-ASP_f1']:.4f}   | {results_without['B-ASP_prec']:.4f}  | {results_without['B-ASP_rec']:.4f}
  Delta                   | {results_with['macro_f1']-results_without['macro_f1']:+.4f}   | {results_with['B-ASP_f1']-results_without['B-ASP_f1']:+.4f}   | {results_with['B-ASP_prec']-results_without['B-ASP_prec']:+.4f}  | {results_with['B-ASP_rec']-results_without['B-ASP_rec']:+.4f}
""")


if __name__ == "__main__":
    main()