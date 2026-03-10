import json
import xml.etree.ElementTree as ET

SYSTEM_PROMPT = """
You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA).
Your task is to identify all explicit aspect terms in each sentence and assign
their sentiment polarity.

An "aspect term" is any product feature, component, service, or entity that
the writer is directly evaluating. Use natural, meaningful noun phrases,
not artificially reduced terms.

Guidelines:

1. Only extract explicit aspects that appear literally in the text.
2. Use natural noun phrases:
   - Good: “battery life”, “customer service”, “build quality”
   - Bad: “life”, “service”, “quality” (too minimal and unclear)
3. Ignore vague nouns by themselves:
   - “problem”, “issue”, “experience”, “thing”, etc.
   unless they refer to a specific concrete target.
4. Do NOT extract aspects that are only implied.
5. Polarity must be one of: "positive", "negative", "neutral".
6. A sentence may contain multiple aspects.
7. If a sentence contains no explicit aspects, return an empty list.
8. The extracted term MUST be a literal substring of the sentence. Do NOT remove or add quotes. Do NOT remove or add quotes.

Return the output in this exact JSON format:

[
  {
    "id": <sentence_id>,
    "sentence": "<text>",
    "aspect_terms": [
        {"term": "...", "polarity": "..."},
        {"term": "...", "polarity": "..."}
    ]
  }
]

If a sentence has no aspects, return:

"aspect_terms": []

"""

def convert_xml(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(output_path, "w", encoding="utf-8") as out:
        for sentence in root.iter("sentence"):
            text_el = sentence.find("text")
            if text_el is None:
                continue

            review_text = text_el.text.strip()

            # Extract aspect terms (labels)
            aspects = []
            aspect_terms_el = sentence.find("aspectTerms")

            if aspect_terms_el is not None:
                for term_el in aspect_terms_el.findall("aspectTerm"):
                    term = term_el.attrib.get("term", "").strip()
                    polarity = term_el.attrib.get("polarity", "").strip()

                    if term and polarity:
                        aspects.append({
                            "term": term,
                            "polarity": polarity
                        })

            # JSON the assistant output
            assistant_json = json.dumps(
                {"aspect_terms": aspects},
                ensure_ascii=False
            )

            # Build training example
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": review_text},
                    {"role": "assistant", "content": assistant_json}
                ]
            }

            out.write(json.dumps(example, ensure_ascii=False) + "\n")


# -------- RUN --------
if __name__ == "__main__":
    convert_xml("Laptop_Train_v2.xml", "train.jsonl")
    print("Done. Output written to train.jsonl")