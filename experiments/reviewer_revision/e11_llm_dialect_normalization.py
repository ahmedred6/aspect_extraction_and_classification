"""Hamzah's review point #3: does the LLM dialect -> MSA normalization help? (protocol D23, fixed before running)

  1. Polarity model: the leakage-safe Arabic model from e02 (word TF-IDF 1-2-grams, 50k features + Logistic
     Regression C=2, normalize_arabic preprocessing), trained on the 80/20 training split (seed 42) only.
  2. Sample: 2,000 reviews, random and stratified by label (seed 42), from the TEST split only.
  3. Dialect identification: zero-shot LLM annotation (the strategy of paper §4.2), Qwen, temperature 0,
     20 reviews per call, labels MSA / DIALECT.
  4. Normalization: every DIALECT review is rewritten once by Qwen (temperature 0) with the pipeline's own
     prompt, MSA_PROMPT, read verbatim from src/6_pipeline/whole_pipeline.py. MSA reviews are not rewritten.
  5. Comparison on the DIALECT subset (same reviews, same model): accuracy and macro-F1, raw vs normalized,
     McNemar exact test. Context: accuracy on the MSA subset.
  6. Human check: 150 random (seed 42) original / normalized pairs exported to CSV.
  7. Every LLM reply is cached in results/e11_cache/, so an interrupted run resumes without repeating calls.

Needs QWEN_API_KEY in .env at the repo root. Run from this folder:
    python e11_llm_dialect_normalization.py --limit 20     # smoke test on 20 reviews
    python e11_llm_dialect_normalization.py                # full run
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _common import REPO, RESULTS_DIR, save_results
from r02_arabic_polarity import normalize_arabic

RS = 42
N_SAMPLE = 2000
BATCH = 20
N_HUMAN = 150
MODEL = "qwen-plus"
URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
CACHE = os.path.join(RESULTS_DIR, "e11_cache")

DIALECT_PROMPT = """You are an expert in Arabic linguistics. For each numbered Arabic review, decide whether it is written
mainly in Modern Standard Arabic (MSA) or mainly in a dialect (Gulf, Egyptian, Levantine, Maghrebi, Iraqi, or
any other colloquial variety). Answer DIALECT if the review contains clear dialectal words or grammar; otherwise MSA.
Return ONLY a JSON list, one object per review, in the same order:
[{"id": <number>, "label": "MSA" or "DIALECT"}, ...]"""


class QuotaExhausted(Exception):
    pass


class Rejected(Exception):
    """The API refused this request (HTTP 400, e.g. its content filter) even after retries."""


# ---------------------------------------------------------------- LLM access
def api_key():
    with open(os.path.join(REPO, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("QWEN_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("QWEN_API_KEY not found in .env")


def chat(messages, max_tokens, key, retries=4):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}",
                                                               "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"], r.get("model", MODEL)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if "insufficient_quota" in msg or "quota" in msg.lower() and e.code in (403, 429):
                raise QuotaExhausted(msg[:300])
            if e.code in (400, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1)); continue
            if e.code == 400:
                raise Rejected(msg[:300])
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1)); continue
            raise


def pipeline_prompt():
    """MSA_PROMPT verbatim from the pipeline source (not imported: the module loads models at import)."""
    src = open(os.path.join(REPO, "src", "6_pipeline", "whole_pipeline.py"), encoding="utf-8").read()
    return re.search(r'MSA_PROMPT = """(.*?)"""', src, flags=re.S).group(1)


def load_cache(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {str(r["id"]): r for r in map(json.loads, f)}


def append_cache(name, row):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- steps
def label_dialect(sample, key, models_seen):
    cache = load_cache("dialect_labels.jsonl")
    todo = [i for i in sample.index if str(i) not in cache]
    print(f"[dialect labels] cached {len(sample) - len(todo)}, to do {len(todo)}", flush=True)
    for b in range(0, len(todo), BATCH):
        ids = todo[b:b + BATCH]
        listing = "\n".join(f"{n + 1}. {sample.at[i, 'text'][:1500]}" for n, i in enumerate(ids))
        try:
            reply, model = chat([{"role": "system", "content": DIALECT_PROMPT},
                                 {"role": "user", "content": listing}], max_tokens=40 * len(ids) + 50, key=key)
        except Rejected:
            print(f"  batch rejected by the API; these {len(ids)} reviews go to the one-by-one pass", flush=True)
            continue
        models_seen.add(model)
        try:
            parsed = json.loads(re.search(r"\[.*\]", reply, flags=re.S).group(0))
            labels = {int(o["id"]): str(o["label"]).upper() for o in parsed}
        except (AttributeError, ValueError, KeyError, TypeError):
            labels = {}
        for n, i in enumerate(ids):
            lab = labels.get(n + 1)
            if lab in ("MSA", "DIALECT"):
                append_cache("dialect_labels.jsonl", {"id": int(i), "label": lab})
        print(f"  labelled {min(b + BATCH, len(todo))}/{len(todo)}", flush=True)
    cache = load_cache("dialect_labels.jsonl")
    missing = [i for i in sample.index if str(i) not in cache]
    print(f"[dialect labels] one-by-one pass for {len(missing)} reviews", flush=True)
    for i in missing:  # one-by-one retry for anything the batch reply did not cover
        try:
            reply, model = chat([{"role": "system", "content": DIALECT_PROMPT},
                                 {"role": "user", "content": f"1. {sample.at[i, 'text'][:1500]}"}], 60, key)
        except Rejected as e:
            append_cache("dialect_labels.jsonl", {"id": int(i), "label": "REJECTED", "reason": str(e)[:200]})
            continue
        m = re.search(r"DIALECT|MSA", reply.upper())
        if m:
            append_cache("dialect_labels.jsonl", {"id": int(i), "label": m.group(0)})
    cache = load_cache("dialect_labels.jsonl")
    return pd.Series({i: cache[str(i)]["label"] for i in sample.index if str(i) in cache})


def normalize_dialect(sample, dialect_ids, key, models_seen):
    prompt = pipeline_prompt()
    cache = load_cache("normalized.jsonl")
    todo = [i for i in dialect_ids if str(i) not in cache]
    print(f"[normalization] cached {len(dialect_ids) - len(todo)}, to do {len(todo)}", flush=True)
    for n, i in enumerate(todo, 1):
        try:
            reply, model = chat([{"role": "system", "content": prompt},
                                 {"role": "user", "content": sample.at[i, "text"]}], max_tokens=3000, key=key)
        except Rejected as e:
            append_cache("normalized.jsonl", {"id": int(i), "normalized": None, "reason": str(e)[:200]})
            continue
        models_seen.add(model)
        append_cache("normalized.jsonl", {"id": int(i), "normalized": reply.strip()})
        if n % 50 == 0 or n == len(todo):
            print(f"  normalized {n}/{len(todo)}", flush=True)
    cache = load_cache("normalized.jsonl")
    return pd.Series({i: cache[str(i)]["normalized"] for i in dialect_ids
                      if str(i) in cache and cache[str(i)]["normalized"]})


def mcnemar(correct_a, correct_b):
    a_only, b_only = int(np.sum(correct_a & ~correct_b)), int(np.sum(~correct_a & correct_b))
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"only_raw_correct": a_only, "only_normalized_correct": b_only, "p_value": float(p)}


def scores(y, pred):
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="use only the first N sampled reviews (smoke test)")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(REPO, "data", "raw", "arabic_100k_reviews.tsv"), sep="\t",
                     names=["label", "text"], header=None)
    df = df[df["label"].isin(["Positive", "Negative"])].reset_index(drop=True)
    y = df["label"].map({"Negative": 0, "Positive": 1})
    tr, te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RS)

    print("[model] training the e02 polarity model on the training split ...", flush=True)
    model = Pipeline([("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000)),
                      ("clf", LogisticRegression(C=2.0, max_iter=2000))])
    model.fit(df.loc[tr, "text"].apply(normalize_arabic), y.loc[tr])
    full_test_acc = accuracy_score(y.loc[te], model.predict(df.loc[te, "text"].apply(normalize_arabic)))
    print(f"  full test accuracy {full_test_acc:.4f} (e02: 0.8738)", flush=True)

    test = df.loc[te]
    sample, _ = train_test_split(test, train_size=N_SAMPLE, stratify=test["label"], random_state=RS)
    if args.limit:
        sample = sample.iloc[:args.limit]
    key, models_seen = api_key(), set()

    try:
        variety = label_dialect(sample, key, models_seen)
        dialect_ids = [i for i in sample.index if variety.get(i) == "DIALECT"]
        normalized = normalize_dialect(sample, dialect_ids, key, models_seen)
    except QuotaExhausted as e:
        print(f"\n[STOPPED] Qwen quota exhausted. Everything so far is cached; rerun to resume.\n{e}")
        sys.exit(2)

    y_s = sample["label"].map({"Negative": 0, "Positive": 1})
    dia = [i for i in dialect_ids if i in normalized.index]
    msa = [i for i in sample.index if variety.get(i) == "MSA"]
    pred_raw = model.predict(sample.loc[dia, "text"].apply(normalize_arabic))
    pred_norm = model.predict(normalized.loc[dia].apply(normalize_arabic))
    yd = y_s.loc[dia].to_numpy()

    res = {"model_requested": MODEL, "models_returned": sorted(models_seen), "run_date": date.today().isoformat(),
           "polarity_model_full_test_accuracy": float(full_test_acc),
           "n_sample": int(len(sample)), "n_labelled": int(len(variety)),
           "variety_counts": variety.value_counts().to_dict(),
           "rejected_by_api": {"labelling": int((variety == "REJECTED").sum()),
                               "normalization": int(len(dialect_ids) - len(dia))},
           "dialect_share": float((variety == "DIALECT").mean()) if len(variety) else None,
           "dialect_raw": scores(yd, pred_raw), "dialect_normalized": scores(yd, pred_norm),
           "mcnemar_raw_vs_normalized": mcnemar(pred_raw == yd, pred_norm == yd),
           "msa_raw": scores(y_s.loc[msa], model.predict(sample.loc[msa, "text"].apply(normalize_arabic)))
           if msa else None,
           "label_balance_dialect": y_s.loc[dia].value_counts().to_dict()}
    for k in ("dialect_raw", "dialect_normalized", "msa_raw"):
        print(f"  {k:20s} {res[k]}")
    print(f"  McNemar raw vs normalized: {res['mcnemar_raw_vs_normalized']}")

    # human check sheet
    rng = random.Random(RS)
    pick = rng.sample(dia, min(N_HUMAN, len(dia)))
    sheet = pd.DataFrame({"id": pick, "gold_label": sample.loc[pick, "label"].values,
                          "original_dialect": sample.loc[pick, "text"].values,
                          "qwen_msa": normalized.loc[pick].values,
                          "meaning_preserved (yes/partly/no)": "", "notes": ""})
    sheet_path = os.path.join(RESULTS_DIR, "e11_human_check.csv")
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    res["human_check_sheet"] = os.path.relpath(sheet_path, REPO)

    if not args.limit:
        save_results("e11_llm_dialect_normalization", res)
    else:
        print("\n(smoke test: results not saved)")


if __name__ == "__main__":
    main()
