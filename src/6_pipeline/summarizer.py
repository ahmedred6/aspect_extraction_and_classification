"""
summarizer.py: guided summarization (paper §3.5, post-processing)

Turns the output of the English and Arabic pipelines into a short
natural-language summary, in three steps:

  1. aggregate_results(): collect the aspect-level results of many reviews
     into one table (language, aspect, positive count, negative count),
     grouping mentions of the same aspect across all reviews.
  2. summarize(): pass ONLY that table to Qwen, which writes a summary of the
     dominant sentiment patterns. The model also returns, as JSON, every
     aspect it mentions with the counts it used.
  3. check_summary(): verify those claims, and every number in the summary
     text, against the table. A summary passes only if nothing disagrees.

Input format: a list of dicts as returned by
`whole_pipeline.analyze_sentence_global()` (src/6_pipeline or src/7_dashboard):
    {"language": "en" | "ar", "overall_polarity": ..., "absa": {"aspects": [
        {"term": ..., "polarity": "positive" | "negative", ...}, ...]}}

The LLM is any callable `llm(messages) -> str`, where messages is a list of
{"role", "content"} dicts. make_qwen_llm() builds the Qwen one with the same
settings the repo already uses (qwen-plus-2025-04-28 via DashScope,
temperature 0); it needs QWEN_API_KEY in a .env file at the repo root.

Run from the repo root:
    python src/6_pipeline/summarizer.py --demo --dry-run     # no API call: shows table + prompt
    python src/6_pipeline/summarizer.py --demo               # calls Qwen
    python src/6_pipeline/summarizer.py --input results.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

QWEN_MODEL = "qwen-plus-2025-04-28"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

POSITIVE, NEGATIVE, MIXED = "positive", "negative", "mixed"


# =====================================================
# 1. AGGREGATION
# =====================================================
def _aspect_key(term: str) -> str:
    """Grouping key: case-folded, whitespace-collapsed aspect term."""
    return re.sub(r"\s+", " ", str(term)).strip().casefold()


def _dominant(pos: int, neg: int) -> str:
    if pos > neg:
        return POSITIVE
    if neg > pos:
        return NEGATIVE
    return MIXED


def aggregate_results(results, top_k=None):
    """
    Aggregate pipeline outputs into aspect-level statistics.

    English and Arabic aspects are kept as separate rows (no translation).
    Returns a dict:
      {"reviews": {...review-level counts...},
       "aspects": [{"language", "aspect", "mentions", "positive",
                    "negative", "dominant"}, ...]}   sorted by mentions
    """
    groups = defaultdict(lambda: {"surface": Counter(), "positive": 0, "negative": 0, "other": 0})
    n_reviews = 0
    n_with_aspects = 0
    overall = Counter()

    for r in results:
        n_reviews += 1
        lang = r.get("language", "en")
        aspects = (r.get("absa") or {}).get("aspects", [])
        if aspects:
            n_with_aspects += 1
        if r.get("overall_polarity") is not None:
            overall[str(r["overall_polarity"]).lower()] += 1

        for asp in aspects:
            key = (lang, _aspect_key(asp["term"]))
            g = groups[key]
            g["surface"][str(asp["term"]).strip()] += 1
            pol = str(asp.get("polarity", "")).lower()
            if pol in (POSITIVE, NEGATIVE):
                g[pol] += 1
            else:
                g["other"] += 1

    rows = []
    for (lang, _), g in groups.items():
        row = {
            "language": lang,
            "aspect": g["surface"].most_common(1)[0][0],
            "mentions": g["positive"] + g["negative"] + g["other"],
            "positive": g["positive"],
            "negative": g["negative"],
            "dominant": _dominant(g["positive"], g["negative"]),
        }
        if g["other"]:
            row["other"] = g["other"]
        rows.append(row)

    rows.sort(key=lambda x: (-x["mentions"], x["language"], x["aspect"]))
    if top_k:
        rows = rows[:top_k]

    return {
        "reviews": {
            "total": n_reviews,
            "with_aspects": n_with_aspects,
            "without_aspects": n_reviews - n_with_aspects,
            "overall_polarity": dict(overall),
        },
        "aspects": rows,
    }


# =====================================================
# 2. GUIDED SUMMARIZATION
# =====================================================
SYSTEM_PROMPT = """You write short business summaries of customer-review statistics.

Rules:
1. Use ONLY the statistics in the table you are given. Do not add aspects, numbers, reasons or opinions that are not in the table.
2. Describe the dominant sentiment patterns: which aspects are mostly praised, which are mostly criticised, and which are mixed.
3. Every number you write must be copied from the table.
4. Write 3 to 6 sentences in English. Arabic aspect names may be written as they appear in the table.

Return ONLY a JSON object, with no other text:
{
  "summary": "<the summary>",
  "claims": [
    {"aspect": "<aspect exactly as in the table>", "language": "<en or ar>",
     "positive": <int>, "negative": <int>, "dominant": "positive" | "negative" | "mixed"}
  ]
}
"claims" must contain one entry for EVERY aspect the summary mentions, with the counts from the table."""


def llm_input(stats):
    """
    What the LLM sees: ONLY the aggregated aspect-sentiment statistics
    (paper §3.5). Review-level counts and the general-polarity results are
    kept for the dashboard but are not passed to the model.
    """
    return {"aspects": [
        {k: a[k] for k in ("language", "aspect", "mentions", "positive", "negative", "dominant")}
        for a in stats["aspects"]
    ]}


def build_messages(stats):
    table = json.dumps(llm_input(stats), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Aspect-sentiment statistics:\n" + table},
    ]


def make_qwen_llm():
    """Qwen client with the repo's existing settings (needs QWEN_API_KEY)."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(".env")
    if not os.getenv("QWEN_API_KEY"):
        raise RuntimeError("QWEN_API_KEY is not set (add it to .env at the repo root)")

    client = ChatOpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL,
        temperature=0,
        max_tokens=3000,
        max_retries=3,
        timeout=None,
    )
    return lambda messages: client.invoke(messages).content


def _parse_json(text):
    """Parse the model's JSON, tolerating ```json fences or text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def summarize(results_or_stats, llm=None, top_k=15):
    """
    Full post-processing step. Accepts pipeline outputs (list) or an
    already-aggregated stats dict. Returns
      {"stats", "summary", "claims", "check", "raw"}.
    """
    if isinstance(results_or_stats, dict) and "aspects" in results_or_stats:
        stats = results_or_stats
    else:
        stats = aggregate_results(results_or_stats, top_k=top_k)

    if not stats["aspects"]:
        return {"stats": stats, "summary": "No aspects were extracted from these reviews.",
                "claims": [], "check": {"passed": True, "errors": [], "note": "nothing to summarize"},
                "raw": None}

    llm = llm or make_qwen_llm()
    raw = llm(build_messages(stats))

    try:
        out = _parse_json(raw)
        summary, claims = str(out.get("summary", "")), out.get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return {"stats": stats, "summary": raw, "claims": [],
                "check": {"passed": False, "errors": ["model did not return valid JSON"]},
                "raw": raw}

    return {"stats": stats, "summary": summary, "claims": claims,
            "check": check_summary(summary, claims, stats), "raw": raw}


# =====================================================
# 3. CONSISTENCY CHECK
# =====================================================
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _allowed_numbers(stats):
    """Every number the summary may legitimately contain: only what the LLM was given."""
    nums = {len(stats["aspects"])}
    for a in stats["aspects"]:
        m, p, n = a["mentions"], a["positive"], a["negative"]
        nums.update([m, p, n])
        if m:
            for share in (p / m, n / m):          # percentages, rounded either way
                nums.update([int(share * 100), round(share * 100), int(share * 100) + 1])
    return nums


def check_summary(summary, claims, stats):
    """
    Verify a summary against the statistics it was generated from.

    Checks:
      a) every claimed aspect exists in the table;
      b) its claimed positive/negative counts and dominant sentiment match;
      c) every number written in the summary text appears in the table
         (counts, review totals, or a percentage derived from them);
      d) every table aspect named in the summary text is listed in the claims.
    """
    errors = []
    index = {(a["language"], _aspect_key(a["aspect"])): a for a in stats["aspects"]}
    by_term = defaultdict(list)
    for (lang, key), a in index.items():
        by_term[key].append(a)

    claimed_keys = set()
    for c in claims:
        key = _aspect_key(c.get("aspect", ""))
        lang = c.get("language")
        row = index.get((lang, key)) if lang else None
        if row is None:
            candidates = by_term.get(key, [])
            row = candidates[0] if len(candidates) == 1 else None
        if row is None:
            errors.append(f"aspect not in the statistics: {c.get('aspect')!r}")
            continue
        claimed_keys.add(key)
        for field in ("positive", "negative"):
            if c.get(field) is not None and int(c[field]) != row[field]:
                errors.append(f"{row['aspect']!r}: {field} count {c[field]} != {row[field]}")
        if c.get("dominant") and str(c["dominant"]).lower() != row["dominant"]:
            errors.append(f"{row['aspect']!r}: dominant {c['dominant']!r} != {row['dominant']!r}")

    allowed = _allowed_numbers(stats)
    text_numbers = [int(x) for x in re.findall(r"\d+", summary.translate(_AR_DIGITS))]
    unverified = sorted({x for x in text_numbers if x not in allowed})
    for x in unverified:
        errors.append(f"number in summary not found in the statistics: {x}")

    lowered = summary.casefold()
    for (lang, key), a in index.items():
        if key and re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", lowered) and key not in claimed_keys:
            errors.append(f"aspect {a['aspect']!r} is mentioned but not listed in the claims")

    return {"passed": not errors, "n_claims": len(claims), "errors": errors}


# =====================================================
# 4. COMMAND LINE
# =====================================================
DEMO_RESULTS = [
    {"language": "en", "overall_polarity": "negative", "absa": {"aspects": [
        {"term": "battery", "polarity": "negative"}, {"term": "screen", "polarity": "positive"}]}},
    {"language": "en", "overall_polarity": "negative", "absa": {"aspects": [
        {"term": "Battery", "polarity": "negative"}, {"term": "price", "polarity": "positive"}]}},
    {"language": "en", "overall_polarity": "positive", "absa": {"aspects": [
        {"term": "screen", "polarity": "positive"}, {"term": "battery", "polarity": "positive"}]}},
    {"language": "en", "overall_polarity": "positive", "absa": {"aspects": []}},
    {"language": "ar", "overall_polarity": "positive", "absa": {"aspects": [
        {"term": "الغرفة", "polarity": "positive"}, {"term": "الخدمة", "polarity": "negative"}]}},
    {"language": "ar", "overall_polarity": "positive", "absa": {"aspects": [
        {"term": "الغرفة", "polarity": "positive"}]}},
]


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Guided aspect-sentiment summarization (§3.5)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSONL file, one analyze_sentence_global() output per line")
    src.add_argument("--demo", action="store_true", help="use a small built-in example")
    ap.add_argument("--dry-run", action="store_true", help="show the table and prompt; do not call the LLM")
    ap.add_argument("--top-k", type=int, default=15, help="number of aspects passed to the LLM")
    args = ap.parse_args()

    results = DEMO_RESULTS if args.demo else _load_jsonl(args.input)
    stats = aggregate_results(results, top_k=args.top_k)

    if args.dry_run:
        for m in build_messages(stats):
            print(f"----- {m['role']} -----\n{m['content']}\n")
        return

    out = summarize(stats)
    print("SUMMARY:\n" + out["summary"] + "\n")
    print("CHECK:", "PASSED" if out["check"]["passed"] else "FAILED")
    for e in out["check"]["errors"]:
        print("  -", e)


if __name__ == "__main__":
    main()
