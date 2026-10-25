"""
xml_to_jsonl.py — step 1 of Arabic aspect extraction
=====================================================
Converts the SemEval-2016 Arabic hotels XML into the JSONL format read by prepare_bio_dataset.py:
    {"id": ..., "sentence": ..., "aspects": [{"aspectTerm": ..., "polarity": ..., "category": ..., "from": ..., "to": ...}]}

(The original conversion script was not in the repo; this reproduces the paper's token counts exactly:
106,876 tokens = 97,235 O / 8,139 B-ASP / 1,502 I-ASP — see experiments/reviewer_revision/r04.)

Run from the repo root:
    python src/3_aspect_extraction/arabic/xml_to_jsonl.py
"""
import json
import os
import xml.etree.ElementTree as ET

INPUT_XML = "data/raw/semeval2016_arabic_hotels_train_v2.xml"
OUTPUT_JSONL = "data/processed/arabic_hotels_train_v2.jsonl"


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    n = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for sent in ET.parse(INPUT_XML).getroot().iter("sentence"):
            text = sent.findtext("text")
            if not text:
                continue
            opinions = sent.find("Opinions")
            aspects = [] if opinions is None else [
                {"aspectTerm": o.get("target"), "polarity": o.get("polarity"),
                 "category": o.get("category"), "from": o.get("from"), "to": o.get("to")}
                for o in opinions.findall("Opinion")]
            out.write(json.dumps({"id": sent.get("id"), "sentence": text, "aspects": aspects},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} sentences to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
