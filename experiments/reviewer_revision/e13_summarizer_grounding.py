"""Is the guided summarizer grounded in its statistics? (protocol D27, fixed before running)

Real reviews with their GOLD aspect labels, so the test isolates the summarizer from extraction errors:
  English: data/raw/amazon_absa_gpt41_labeled.jsonl
  Arabic : SemEval-2016 Arabic hotels (NULL aspects dropped)
Positive / negative aspects only (the pipeline is binary). 25 batches per language, each of 30 random reviews
with at least one aspect (seed 42). Every batch goes through src/6_pipeline/summarizer.py unchanged:
aggregation -> Qwen (qwen-plus, temperature 0) -> consistency check.
Reported: share of summaries that pass the check, and every error by type.
"""
import json
import os
import random
import sys
from collections import Counter

from _common import REPO, save_results

sys.path.insert(0, os.path.join(REPO, "src", "6_pipeline"))
os.chdir(REPO)  # summarizer reads .env from the repo root
import summarizer  # noqa: E402

SEED, N_BATCHES, BATCH_SIZE = 42, 25, 30
POLAR = {"positive", "negative"}


def english_reviews():
    out = []
    for line in open(os.path.join(REPO, "data", "raw", "amazon_absa_gpt41_labeled.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        asp = [{"term": a["term"], "polarity": a["polarity"]} for a in r.get("aspect_terms", [])
               if a.get("polarity") in POLAR]
        if asp:
            out.append({"language": "en", "absa": {"aspects": asp}})
    return out


def arabic_reviews():
    out = []
    path = os.path.join(REPO, "experiments", "reviewer_revision", "data", "Arabic_Hotels_TrD_V2.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        asp = [{"term": a["aspectTerm"], "polarity": a["polarity"]} for a in r.get("aspects", [])
               if a.get("aspectTerm") not in (None, "", "NULL") and a.get("polarity") in POLAR]
        if asp:
            out.append({"language": "ar", "absa": {"aspects": asp}})
    return out


def error_type(e):
    for key, label in (("not in the statistics", "invented aspect"), ("count", "wrong count"),
                       ("dominant", "flipped sentiment"), ("number in summary", "unsupported number"),
                       ("not listed in the claims", "mentioned but not claimed"), ("valid JSON", "invalid JSON")):
        if key in e:
            return label
    return "other"


def main():
    llm = summarizer.make_qwen_llm()
    rng = random.Random(SEED)
    res = {"model": summarizer.QWEN_MODEL, "batches_per_language": N_BATCHES, "reviews_per_batch": BATCH_SIZE}
    examples = []
    for lang, reviews in (("english", english_reviews()), ("arabic", arabic_reviews())):
        runs, errors = [], Counter()
        for b in range(N_BATCHES):
            out = summarizer.summarize(rng.sample(reviews, BATCH_SIZE), llm=llm)
            chk = out["check"]
            for e in chk["errors"]:
                errors[error_type(e)] += 1
            runs.append({"passed": chk["passed"], "n_claims": chk.get("n_claims"), "errors": chk["errors"],
                         "n_aspects_in_table": len(out["stats"]["aspects"]),
                         "summary": out["summary"], "claims": out["claims"]})
            if b < 2:
                examples.append({"language": lang, "summary": out["summary"], "check": chk})
            print(f"  {lang} batch {b + 1:2d}: {'PASS' if chk['passed'] else 'FAIL ' + '; '.join(chk['errors'])}",
                  flush=True)
        passed = sum(r["passed"] for r in runs)
        res[lang] = {"n_reviews_available": len(reviews), "passed": passed, "total": len(runs),
                     "pass_rate": passed / len(runs), "error_types": dict(errors),
                     "mean_claims": sum(r["n_claims"] or 0 for r in runs) / len(runs), "runs": runs}
        print(f"### {lang}: {passed}/{len(runs)} summaries passed; errors {dict(errors)}", flush=True)
    res["examples"] = examples
    # run 1 (D27) is kept as e13_summarizer_grounding; run 2 (D27 amendment) is saved separately
    save_results("e13_summarizer_grounding" + (f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""), res)


if __name__ == "__main__":
    main()
