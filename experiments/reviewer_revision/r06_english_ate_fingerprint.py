"""Identify the data behind paper §6.2 / Fig. 3 (English aspect extraction) by token counts.

Fig. 3 (aspect vs non-aspect confusion matrix) covers 101,005 tokens: 93,296 non-aspect + 7,709 aspect.
§4.3 says the English ABSA data is SemEval-2014 Task 4 Laptop + Restaurant (">6,000 sentences").
This script counts tokens / aspect tokens for each domain combination and each BIO-preparation variant of
english/prepare_dataset_en.py, to find which one gives exactly 101,005 / 7,709.
Restaurant data: data/raw/semeval2014_restaurants_train_v2.xml (official v2, Hugging Face mirror).
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from itertools import product

import spacy

from _common import REPO, load_original, save_results

NLP = spacy.blank("en")  # tokenizer only (en_core_web_sm fails to load on this Windows build; same rules)

FILES = {"laptop": os.path.join(REPO, "data", "raw", "semeval2014_laptops_train_v2.xml"),
         "restaurant": os.path.join(REPO, "data", "raw", "semeval2014_restaurants_train_v2.xml")}
TARGET = {"tokens": 101005, "aspect_tokens": 7709}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

norm = load_original("src/3_aspect_extraction/english/normalizer_en.py", "normalizer_en").normalize_english


def load(domain):
    rows = []
    for s in ET.parse(FILES[domain]).getroot().findall("sentence"):
        t = s.find("text")
        if t is None or not t.text:
            continue
        terms = [] if s.find("aspectTerms") is None else [
            {"term": a.get("term"), "polarity": a.get("polarity"), "from": int(a.get("from")), "to": int(a.get("to"))}
            for a in s.find("aspectTerms").findall("aspectTerm")]
        rows.append({"sentence": t.text.strip(), "aspect_terms": terms})
    return rows


def bio(rows, normalize, fixed_char_map, polarities, dedupe, tokenizer="regex"):
    """english/prepare_dataset_en.py, with switches for the variants."""
    total = aspect = 0
    seen = set()
    for item in rows:
        sentence = norm(item["sentence"]) if normalize else item["sentence"]
        if dedupe:  # load_dataset() in training_with_grid_search_en.py groups tokens by sentence string
            if sentence in seen:
                continue
            seen.add(sentence)
        if tokenizer == "spacy":  # §6.2.1: spaCy tokenizer used for preprocessing; offsets are exact
            doc = NLP(sentence)
            tokens = [t.text for t in doc]
            char_map = [(t.idx, t.idx + len(t.text)) for t in doc]
        elif fixed_char_map:
            tokens, char_map = [], []
            for m in TOKEN_RE.finditer(sentence):
                tokens.append(m.group()); char_map.append((m.start(), m.end()))
        else:  # original: assumes exactly one space between tokens
            tokens = TOKEN_RE.findall(sentence)
            char_map, idx = [], 0
            for t in tokens:
                char_map.append((idx, idx + len(t))); idx += len(t) + 1
        labels = ["O"] * len(tokens)
        for a in item["aspect_terms"]:
            if polarities and a["polarity"] not in polarities:
                continue
            inside = False
            for i, (cs, ce) in enumerate(char_map):
                if cs >= a["from"] and ce <= a["to"]:
                    labels[i] = "I-ASP" if inside else "B-ASP"
                    inside = True
        total += len(tokens)
        aspect += sum(l != "O" for l in labels)
    return total, aspect


def main():
    data = {d: load(d) for d in FILES}
    results = []
    pol_options = {"pos/neg": ("positive", "negative"), "pos/neg/neu": ("positive", "negative", "neutral"), "all": None}
    for domains in (["laptop"], ["restaurant"], ["laptop", "restaurant"]):
        rows = [r for d in domains for r in data[d]]
        for tokenizer, normalize, fixed, pol_name, dedupe in product(
                ("regex", "spacy"), (True, False), (False, True), pol_options, (True, False)):
            if tokenizer == "spacy" and not fixed:
                continue  # spaCy gives exact offsets; the char_map switch only applies to the regex tokenizer
            total, aspect = bio(rows, normalize, fixed, pol_options[pol_name], dedupe, tokenizer)
            r = {"domains": "+".join(domains), "sentences": len(rows), "tokenizer": tokenizer,
                 "normalize_english": normalize, "char_map": "fixed" if fixed else "original",
                 "polarities": pol_name, "dedupe_sentences": dedupe, "tokens": total, "aspect_tokens": aspect,
                 "exact_match": total == TARGET["tokens"] and aspect == TARGET["aspect_tokens"]}
            results.append(r)
    results.sort(key=lambda r: abs(r["tokens"] - TARGET["tokens"]) + abs(r["aspect_tokens"] - TARGET["aspect_tokens"]))
    for r in results[:10]:
        print(json.dumps(r))
    save_results("r06_english_ate_fingerprint", {"variants": results}, paper={"fig3": TARGET})


if __name__ == "__main__":
    main()
