"""
xml_to_jsonl.py — step 1 of English aspect extraction
======================================================
Converts SemEval-2014 Task 4 XML into the JSONL format read by prepare_dataset_en.py:
    {"id": ..., "sentence": ..., "aspect_terms": [{"term": ..., "polarity": ..., "from": ..., "to": ...}]}

By default it converts the Laptop train set (what prepare_dataset_en.py was written for).
The paper's Table 3 / Fig. 3 numbers correspond to Laptop + Restaurant combined (see
experiments/reviewer_revision/RESULTS_SUMMARY.md §6); pass --with-restaurant to include it.

Run from the repo root:
    python src/3_aspect_extraction/english/xml_to_jsonl.py [--with-restaurant]
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

LAPTOP_XML = "data/raw/semeval2014_laptops_train_v2.xml"
RESTAURANT_XML = "data/raw/semeval2014_restaurants_train_v2.xml"
OUTPUT_JSONL = "data/processed/laptop_train_v2.jsonl"


def convert(xml_path, out):
    n = 0
    for s in ET.parse(xml_path).getroot().findall("sentence"):
        t = s.find("text")
        if t is None or not t.text:
            continue
        terms = [] if s.find("aspectTerms") is None else [
            {"term": a.get("term"), "polarity": a.get("polarity"),
             "from": int(a.get("from")), "to": int(a.get("to"))}
            for a in s.find("aspectTerms").findall("aspectTerm")]
        out.write(json.dumps({"id": s.get("id"), "sentence": t.text.strip(), "aspect_terms": terms},
                             ensure_ascii=False) + "\n")
        n += 1
    return n


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    sources = [LAPTOP_XML] + ([RESTAURANT_XML] if "--with-restaurant" in sys.argv else [])
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for src in sources:
            print(f"{src}: {convert(src, out)} sentences")
    print(f"Wrote {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
