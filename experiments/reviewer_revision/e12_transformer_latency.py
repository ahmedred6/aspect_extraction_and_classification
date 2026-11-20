"""Hamzah's review point #5: our classical ABSA pipeline vs. a transformer-equivalent pipeline on the SAME CPU
(protocol D25, fixed before running).

Same reviews and protocol as e03 (105 per language, seed 42, first 5 warm-up excluded, batch size 1, CPU).
Per review, both pipelines do the same work:
  ours         aspect extraction (SVM BIO tagger) + one classification per extracted aspect (stacking ensemble)
  transformer  one token-classification forward pass over the review (extraction)
               + one sequence-classification forward pass per aspect, on the same aspect windows
The transformer classifies exactly the aspects our extractor found, so the workload is identical.
Tokenization is timed. Heads are randomly initialized: inference cost does not depend on the weights, so only
latency is reported. Language detection, routing and the LLM call are excluded on both sides.
Threads: single-threaded for both (primary); transformers also with all CPU threads (their best case).

This laptop is NOT the Table 15 machine; Table 15 is unchanged. This is a separate same-CPU comparison.
"""
import json
import os
import random
import time

import e03_latency_benchmark as e03  # sets OMP/MKL/OpenBLAS to 1 thread before numpy loads

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from _common import REPO, save_results

MODELS = {"en": ["bert-base-uncased", "distilbert-base-uncased"],
          "ar": ["aubmindlab/bert-base-arabertv02"]}


def stats(ms):
    a = np.asarray(ms[e03.N_WARMUP:])
    return {"mean_ms": float(a.mean()), "std_ms": float(a.std()), "p95_ms": float(np.percentile(a, 95))}


def time_ours(sentences, m, lang):
    """ABSA stages of the classical pipeline; also returns the aspect windows for the transformer."""
    totals, work = [], []
    for s in sentences:
        t = time.perf_counter()
        norm, aspects = e03.extract(s, m, lang)
        e03.classify(norm, aspects, m, lang)
        totals.append((time.perf_counter() - t) * 1000)
        windows = [e03.ar_pipe.char_to_token_window(norm, a["from"], a["to"], window_size=5) for a in aspects]
        work.append((s, [(w, a["term"]) for w, a in zip(windows, aspects)]))
    return stats(totals), work


def time_transformer(name, work):
    tok = AutoTokenizer.from_pretrained(name)
    tagger = AutoModelForTokenClassification.from_pretrained(name, num_labels=3).eval()
    clf = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).eval()
    totals = []
    with torch.inference_mode():
        for sentence, aspects in work:
            t = time.perf_counter()
            tagger(**tok(sentence, return_tensors="pt", truncation=True, max_length=512))
            for window, term in aspects:
                clf(**tok(window, term, return_tensors="pt", truncation=True, max_length=512))
            totals.append((time.perf_counter() - t) * 1000)
    params = sum(p.numel() for p in clf.parameters())
    return stats(totals), int(params)


def main():
    hw = e03.hardware()
    print(hw, flush=True)
    m = e03.train_models()
    rng = random.Random(e03.SEED)  # identical sampling to e03.main()
    ar_s = [json.loads(l)["sentence"] for l in open(os.path.join(e03.DATA, "Arabic_Hotels_TrD_V2.jsonl"), encoding="utf-8")]
    en_s = [json.loads(l)["sentence"] for l in open(os.path.join(REPO, "data", "raw", "amazon_absa_gpt41_labeled.jsonl"), encoding="utf-8")]
    samples = {"ar": rng.choices(ar_s, k=e03.N_REVIEWS + e03.N_WARMUP),
               "en": rng.choices(en_s, k=e03.N_REVIEWS + e03.N_WARMUP)}

    all_threads = os.cpu_count()
    res = {"hardware": hw, "protocol": {"reviews": e03.N_REVIEWS + e03.N_WARMUP, "warmup_excluded": e03.N_WARMUP,
                                        "batch_size": 1, "all_threads": all_threads,
                                        "torch": torch.__version__}}
    for lang in ("en", "ar"):
        ours, work = time_ours(samples[lang], m, lang)
        n_aspects = float(np.mean([len(a) for _, a in work[e03.N_WARMUP:]]))
        out = {"ours_absa_stages": ours, "mean_aspects_per_review": n_aspects, "transformers": {}}
        print(f"\n### {lang}: ours {ours['mean_ms']:.2f} ms/review ({n_aspects:.2f} aspects/review)", flush=True)
        for name in MODELS[lang]:
            row = {}
            for label, threads in (("1_thread", 1), ("all_threads", all_threads)):
                torch.set_num_threads(threads)
                s, params = time_transformer(name, work)
                row[label] = {**s, "threads": torch.get_num_threads(),
                              "slowdown_vs_ours": s["mean_ms"] / ours["mean_ms"]}
                print(f"  {name:34s} {label:11s} {s['mean_ms']:8.1f} ms/review  "
                      f"= {row[label]['slowdown_vs_ours']:.0f}x ours", flush=True)
            row["parameters"] = params
            out["transformers"][name] = row
        res[{"en": "english", "ar": "arabic"}[lang]] = out
    save_results("e12_transformer_latency", res)


if __name__ == "__main__":
    main()
