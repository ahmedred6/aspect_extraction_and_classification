import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

# ===========================
# Pydantic Schemas
# ===========================

class Aspect(BaseModel):
    term: str
    polarity: Literal["positive", "neutral", "negative"]

class Review(BaseModel):
    id: int
    sentence: str
    aspect_terms: List[Aspect]

class Reviews(BaseModel):
    data: List[Review]


# ===========================
# Load API Key
# ===========================

load_dotenv(".env")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    max_retries=3,
    timeout=None
)

llm = llm.with_structured_output(Reviews, strict=True)


# ===========================
# Config
# ===========================

INPUT_FILE = "amazon_reviews.txt"
OUTPUT_FILE = "aspect_results.jsonl"
CHECKPOINT_FILE = "checkpoint.txt"
BATCH_SIZE = 20


# ===========================
# Prompt Instructions
# ===========================

INSTRUCTIONS = """
You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA).
Your task is to identify all explicit aspect terms in each sentence and assign
their sentiment polarity.

An "aspect term" is any product feature, component, service, or entity that
the writer is directly evaluating. Use natural, meaningful noun phrases,
not artificially reduced terms.

Guidelines:

1. Extract ONLY explicit features that are concrete and directly evaluated in the sentence.
2. Ignore vague nouns by themselves:
   - “problem”, “issue”, “experience”, “thing”, etc.
   unless they refer to a specific concrete target.
3. Do NOT extract aspects that are only implied.
4. Remove adjectives or sentiment words from aspect terms.
   Extract the minimal noun phrase that identifies the feature.
   (Example: “excellent support service” → “support service”)
5. A sentence may contain multiple aspects.
6. If a sentence contains no explicit aspects, return an empty list.
7. The extracted term MUST be a literal substring of the sentence. Do NOT remove or add quotes. Do NOT remove or add quotes.

Return the output in this exact JSON format
"""

# ===========================
# Utility: Compute Offsets
# ===========================

def add_offsets(sentence: str, aspect_list: List[dict]):
    for asp in aspect_list:
        term = asp["term"]
        idx = sentence.lower().find(term.lower())
        if idx == -1:
            asp["from"] = -1
            asp["to"] = -1
        else:
            asp["from"] = idx
            asp["to"] = idx + len(term)
    return aspect_list


# ===========================
# Load Sentences
# ===========================

def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ===========================
# Extract Batch (Safe Mode)
# ===========================

def extract_batch_safe(sentences_batch, global_offset):
    numbered = [
        {"id": global_offset + i + 1, "sentence": s}
        for i, s in enumerate(sentences_batch)
    ]

    prompt = INSTRUCTIONS + "\n\nSentences:\n" + json.dumps(numbered, ensure_ascii=False)

    try:
        parsed = llm.invoke(prompt)
    except Exception as e:
        print(f"❌ LLM failed on batch starting at {global_offset}: {e}")
        return None  # skip batch safely

    results = []
    for r in parsed.data:
        entry = r.model_dump()
        entry["aspect_terms"] = add_offsets(entry["sentence"], entry["aspect_terms"])
        results.append(entry)

    return results


# ===========================
# Main Execution (Fault-Tolerant)
# ===========================

def main():
    # Load sentences
    sentences = load_sentences(INPUT_FILE)
    total = len(sentences)
    print(f"Loaded {total} sentences.")

    # Load checkpoint (resume safely)
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as ck:
            start_index = int(ck.read().strip())
        print(f"🔄 Resuming from sentence {start_index}...")
    else:
        start_index = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for start in range(start_index, total, BATCH_SIZE):
            batch = sentences[start : start + BATCH_SIZE]

            print(f"Processing batch {start} → {start + len(batch)}...")

            results = extract_batch_safe(batch, global_offset=start)

            if results is None:
                print("⚠ Skipping batch due to failure.")
            else:
                for r in results:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()  # MAKE SURE it's physically written to disk

            # Update checkpoint
            with open(CHECKPOINT_FILE, "w") as ck:
                ck.write(str(start + BATCH_SIZE))

    print("✔ Completed. Results saved to:", OUTPUT_FILE)
    print("✔ Checkpoint saved to:", CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
