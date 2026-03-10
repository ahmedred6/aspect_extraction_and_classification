import os
import json
from time import sleep
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

class Aspect(BaseModel):
    term: str = Field(description="Product Feature")
    polarity: Literal["positive", "neutral", "negative"]

class Review(BaseModel):
    id: int = Field(description="ID of the review")
    sentence: str = Field(description="The sentence itself")
    aspect_terms: List[Aspect]



# ===========================
# Load API Key
# ===========================
load_dotenv(".env")
"""
llm = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    model="qwen-plus-2025-04-28",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    max_retries=3,
    timeout=None
)
"""
#ft:gpt-4o-mini-2024-07-18:personal :: ChVGIo8P

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",      # or "gpt-4o" or "gpt-4o-mini"
    max_retries=3,
    timeout=None
)
# ===========================
# Config
# ===========================
INPUT_FILE = "sentences.txt"
OUTPUT_FILE = "aspect_results.jsonl"
BATCH_SIZE = 20


# ===========================
# Universal ABSA Prompt
# ===========================
INSTRUCTIONS = """
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


# ===========================
# Utility: Compute Offsets
# ===========================
def add_offsets(sentence, aspects):
    """
    Adds 'from' and 'to' offsets to each aspect term.
    """
    for asp in aspects:
        term = asp["term"]

        # literal match
        idx = sentence.lower().find(term.lower())

        if idx == -1:
            asp["from"] = -1
            asp["to"] = -1
        else:
            asp["from"] = idx
            asp["to"] = idx + len(term)

    return aspects


# ===========================
# Load Sentences
# ===========================
def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ===========================
# Extract Batch of Sentences
# ===========================
def extract_batch(sentences_batch, global_offset):
    numbered = [
        {"id": global_offset + i + 1, "sentence": s}
        for i, s in enumerate(sentences_batch)
    ]

    batch_json = json.dumps(numbered, ensure_ascii=False, indent=2)
    prompt = INSTRUCTIONS + "\n\nSentences:\n" + batch_json
    llm_with_reviews = llm.bind_tools([Reviews])
    response = llm_with_reviews.invoke(prompt)
    content = response.content.strip()

    # Parse response JSON
    try:
        parsed = json.loads(content)
    except:
        json_text = content[content.find("[") : content.rfind("]") + 1]
        parsed = json.loads(json_text)

    # Add offsets for each sentence
    for entry in parsed:
        entry["aspect_terms"] = add_offsets(entry["sentence"], entry["aspect_terms"])

    return parsed


# ===========================
# Save Results
# ===========================
def save_to_jsonl(results, writer):
    for entry in results:
        writer.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ===========================
# Main Pipeline
# ===========================
def main():
    sentences = load_sentences(INPUT_FILE)
    total = len(sentences)

    print(f"Loaded {total} sentences.")
    print("Starting batch processing...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as writer:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = sentences[batch_start : batch_start + BATCH_SIZE]
            batch_idx = batch_start // BATCH_SIZE + 1

            print(f"Processing batch {batch_idx} ({len(batch)} sentences)...")

            try:
                results = extract_batch(batch, global_offset=batch_start)
            except Exception as e:
                print("Batch failed:", e)
                continue

            save_to_jsonl(results, writer)
            sleep(0.2)

    print("Done! Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
