"""
fix_bio_tagging.py
==================
Drop-in replacement for prepare_dataset.py with two fixes:

FIX 1 — char_map built from actual regex match offsets, not by counting.
         The original code assumed exactly one space between every token
         (idx += len(t) + 1), which diverged from sentence.find() positions
         and caused ALL multi-token aspects to miss I-ASP assignment.

FIX 2 — First matched token in a span → B-ASP, subsequent → I-ASP.
         The logic was already in the original code but never triggered
         because of Fix 1's bug.

Run from the project root:
    python fix_bio_tagging.py

Outputs:
    arabic_ate_features_fixed.json   (drop-in replacement for arabic_ate_features.json)
"""

import json
import re
import os
from collections import Counter
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE  = "Arabic_Hotels_TrD_V2.jsonl"
OUTPUT_FILE = "arabic_ate_features_fixed.json"

# Adjust if running from a different working directory
if not os.path.exists(INPUT_FILE):
    INPUT_FILE  = "Hamzah`s_Part/Arabic_Hotels_TrD_V2.jsonl"
    OUTPUT_FILE = "Hamzah`s_Part/arabic_ate_features_fixed.json"


# ── normalizer (same logic as normalizer.py) ───────────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)   # diacritics
    text = re.sub(r"ـ", "", text)                               # tatweel
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ي", "ي").replace("ك", "ك")
    text = re.sub(r"[^\w\s،؟.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── tokeniser ─────────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text)


# ── FIX 1: build char_map from actual match offsets ───────────────────────────
def build_char_map(text):
    """
    Returns (tokens, char_map) where char_map[i] = (start, end) in `text`.
    Uses re.finditer so positions are guaranteed to align with text.find().
    """
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


# ── main ──────────────────────────────────────────────────────────────────────
def prepare_dataset():
    samples = []
    label_counter = Counter()
    fixed_iasp = 0          # count of I-ASP tags that are new

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing sentences"):
        item = json.loads(line)

        raw_sentence = item["sentence"]
        sentence     = normalize_arabic(raw_sentence)

        # FIX 1: char_map from actual positions
        tokens, char_map = build_char_map(sentence)
        labels = ["O"] * len(tokens)

        filtered_aspects = [
            asp for asp in item["aspects"]
            if asp["polarity"] in ["positive", "negative"]
        ]

        for asp in filtered_aspects:
            aspect_term = normalize_arabic(asp["aspectTerm"])
            occurrences = find_all_occurrences(sentence, aspect_term)

            if not occurrences:
                continue

            for (start, end) in occurrences:
                # FIX 2: first token in span → B-ASP, rest → I-ASP
                inside = False
                for i, (cstart, cend) in enumerate(char_map):
                    if cstart >= start and cend <= end:
                        if not inside:
                            labels[i] = "B-ASP"
                            inside = True
                        else:
                            labels[i] = "I-ASP"
                            fixed_iasp += 1

        for tok, lab, (cstart, cend) in zip(tokens, labels, char_map):
            samples.append({
                "token":    tok,
                "label":    lab,
                "sentence": sentence
            })
            label_counter[lab] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n✔  Saved {len(samples)} token samples → {OUTPUT_FILE}")
    print(f"\nLabel distribution:")
    for lbl, cnt in sorted(label_counter.items()):
        print(f"   {lbl:8s}: {cnt:>7,}")
    print(f"\nNew I-ASP tags added vs. original: {fixed_iasp}")
    print("\nNext step: re-run train_svm_ate.py (or evaluate_svm_split.py)")
    print("pointing FEATURES_FILE at:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_dataset()