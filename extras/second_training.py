import xml.etree.ElementTree as ET
import json
import random

# ================================================================
# CONFIG
# ================================================================
XML_FILE = "Laptop_Train_v2.xml"
OUTPUT_JSONL = "laptop_batched_train.jsonl"

# How many sentences per batch (model must learn multiple patterns)
BATCH_SIZES = [2, 3, 4, 5, 6]  
EXAMPLES_PER_BATCH_SIZE = 15    # total ≈ 75 training examples


# ================================================================
# 1) Parse the SemEval XML
# ================================================================
def load_sentences_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    dataset = []
    for sentence in root.iter("sentence"):
        sent_id = sentence.attrib["id"]
        text = sentence.find("text").text

        aspects_list = []
        opinions = sentence.find("aspectTerms")
        if opinions is not None:
            for op in opinions.iter("aspectTerm"):
                aspects_list.append({
                    "term": op.attrib["term"],
                    "polarity": op.attrib["polarity"]
                })

        dataset.append({
            "id": sent_id,
            "sentence": text,
            "aspect_terms": aspects_list
        })

    return dataset


# ================================================================
# 2) Build training samples in OpenAI FT format
# ================================================================
def make_training_example(batch_items):

    # System prompt (same as your fine-tune)
    system_prompt = """
You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA).
Your task is to identify all explicit aspect terms in each sentence and assign
their sentiment polarity.

Return exactly one JSON array called "results", with one item per sentence.
"""

    # Build user input JSON
    user_batch = {
        "batch": [
            {
                "id": item["id"],
                "sentence": item["sentence"]
            }
            for item in batch_items
        ]
    }

    # Build assistant output JSON array
    assistant_results = []
    for item in batch_items:
        assistant_results.append({
            "id": item["id"],
            "sentence": item["sentence"],
            "aspect_terms": item["aspect_terms"]  # already structured
        })

    # Final fine-tuning example
    example = {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": json.dumps(user_batch, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"results": assistant_results}, ensure_ascii=False)}
        ]
    }

    return example


# ================================================================
# 3) Build batched training dataset
# ================================================================
def build_batched_dataset(data):
    examples = []

    for batch_size in BATCH_SIZES:
        for _ in range(EXAMPLES_PER_BATCH_SIZE):
            sampled = random.sample(data, batch_size)
            example = make_training_example(sampled)
            examples.append(example)

    return examples


# ================================================================
# 4) Save JSONL
# ================================================================
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("Loading XML...")
    data = load_sentences_from_xml(XML_FILE)
    print(f"Loaded {len(data)} sentences.")

    print("Building batched training examples...")
    examples = build_batched_dataset(data)
    print(f"Generated {len(examples)} fine-tuning samples.")

    print(f"Saving to {OUTPUT_JSONL}...")
    save_jsonl(examples, OUTPUT_JSONL)

    print("Done! →", OUTPUT_JSONL)
