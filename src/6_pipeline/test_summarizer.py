"""
test_summarizer.py: offline tests for summarizer.py (no API key needed).

Stand-in LLMs return fixed answers: one faithful summary, and one for each
kind of mistake the consistency check must catch.

Run from the repo root:
    python src/6_pipeline/test_summarizer.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarizer import DEMO_RESULTS, aggregate_results, build_messages, summarize  # noqa: E402


def fake_llm(summary, claims):
    return lambda messages: json.dumps({"summary": summary, "claims": claims}, ensure_ascii=False)


GOOD_CLAIMS = [
    {"aspect": "battery", "language": "en", "positive": 1, "negative": 2, "dominant": "negative"},
    {"aspect": "screen", "language": "en", "positive": 2, "negative": 0, "dominant": "positive"},
    {"aspect": "الغرفة", "language": "ar", "positive": 2, "negative": 0, "dominant": "positive"},
]
GOOD_SUMMARY = ("The battery is the most criticised aspect, with 2 negative "
                "and 1 positive mentions (67% negative). The screen is consistently praised "
                "(2 positive mentions). Arabic reviews praise الغرفة in 2 mentions.")


def run():
    stats = aggregate_results(DEMO_RESULTS)
    rows = {(a["language"], a["aspect"].casefold()): a for a in stats["aspects"]}
    failures = []

    def expect(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name)
        if not cond:
            failures.append(name)

    # --- aggregation ---
    expect("groups 'battery' and 'Battery' into one aspect", rows[("en", "battery")]["mentions"] == 3)
    expect("battery counts are 1 positive / 2 negative",
           (rows[("en", "battery")]["positive"], rows[("en", "battery")]["negative"]) == (1, 2))
    expect("battery is dominant negative", rows[("en", "battery")]["dominant"] == "negative")
    expect("English and Arabic aspects kept separate", ("ar", "الغرفة") in rows)
    expect("review totals: 6 reviews, 5 with aspects, 1 without",
           (stats["reviews"]["total"], stats["reviews"]["with_aspects"],
            stats["reviews"]["without_aspects"]) == (6, 5, 1))
    expect("aspects sorted by mentions", stats["aspects"][0]["aspect"].casefold() == "battery")

    # --- prompt contains only the table ---
    msgs = build_messages(stats)
    expect("prompt carries the statistics as JSON", '"battery"' in msgs[1]["content"].casefold())
    expect("prompt carries no review text", "review text" not in msgs[1]["content"])
    expect("prompt carries only aspect statistics (no review-level or general-polarity counts)",
           "overall_polarity" not in msgs[1]["content"] and "with_aspects" not in msgs[1]["content"])

    out = summarize(DEMO_RESULTS, llm=fake_llm("Across 6 reviews, " + GOOD_SUMMARY, GOOD_CLAIMS))
    expect("a number the LLM was not given (review total) is caught",
           any("number in summary not found" in e and e.endswith(" 6") for e in out["check"]["errors"]))

    # --- check: faithful summary passes ---
    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY, GOOD_CLAIMS))
    expect("faithful summary passes the check", out["check"]["passed"])
    if not out["check"]["passed"]:
        print("      ", out["check"]["errors"])

    # --- check: each kind of mistake is caught ---
    bad_count = [dict(GOOD_CLAIMS[0], negative=5)] + GOOD_CLAIMS[1:]
    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY, bad_count))
    expect("wrong count is caught", not out["check"]["passed"]
           and any("negative count 5" in e for e in out["check"]["errors"]))

    invented = GOOD_CLAIMS + [{"aspect": "camera", "language": "en", "positive": 3, "negative": 0}]
    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY + " The camera is loved.", invented))
    expect("invented aspect is caught", any("not in the statistics" in e for e in out["check"]["errors"]))

    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY.replace("2 negative", "9 negative"), GOOD_CLAIMS))
    expect("invented number in the text is caught", any("number in summary" in e for e in out["check"]["errors"]))

    flipped = [dict(GOOD_CLAIMS[1], dominant="negative")] + [GOOD_CLAIMS[0], GOOD_CLAIMS[2]]
    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY, flipped))
    expect("flipped sentiment is caught", any("dominant" in e for e in out["check"]["errors"]))

    out = summarize(DEMO_RESULTS, llm=fake_llm(GOOD_SUMMARY + " Price is liked too.", GOOD_CLAIMS))
    expect("aspect mentioned but not claimed is caught",
           any("'price' is mentioned" in e for e in out["check"]["errors"]))

    out = summarize(DEMO_RESULTS, llm=lambda m: "Sure! The battery is bad.")
    expect("non-JSON answer fails the check", not out["check"]["passed"])

    fenced = "```json\n" + json.dumps({"summary": GOOD_SUMMARY, "claims": GOOD_CLAIMS}, ensure_ascii=False) + "\n```"
    out = summarize(DEMO_RESULTS, llm=lambda m: fenced)
    expect("JSON inside ``` fences is accepted", out["check"]["passed"])

    out = summarize([{"language": "en", "absa": {"aspects": []}}], llm=lambda m: 1 / 0)
    expect("no aspects: no LLM call, clear message", "No aspects" in out["summary"])

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
