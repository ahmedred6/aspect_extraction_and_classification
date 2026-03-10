import json
import os
from typing import List, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ============================
# Config
# ============================

INPUT_FILE = "amazon_reviews_arabic.jsonl"
OUTPUT_FILE = "arabic_classification.jsonl"

BATCH_SIZE = 20
MAX_SENTENCES = 5000  # <-- change this to limit how many sentences you label (None for all)


# ============================
# 1. Pydantic Schemas
# ============================

class SentenceClassification(BaseModel):
    index: int = Field(
        description="Index of the sentence in the batch (0-based)."
    )
    label: Literal["positive", "negative"] = Field(
        description='Sentiment label: "positive" or "negative". No neutral.'
    )


class BatchClassification(BaseModel):
    results: List[SentenceClassification]


# ============================
# 2. LLM Init with Structured Output
# ============================

load_dotenv(".env")

llm_raw = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-04-28",
    temperature=0,
    max_tokens=3000,
)

# Force the LLM to return a BatchClassification object
llm = llm_raw.with_structured_output(BatchClassification, strict=True)


# ============================
# 3. System Prompt
# ============================

SYSTEM_PROMPT = """
You are an expert Arabic sentiment classifier.

Your task:
- For each sentence, classify the sentiment as either:
  - "positive"
  - "negative"

Rules:
- Do NOT use "neutral". Force a decision: positive or negative.
- Very short praise/compliment/recommendation → positive.
- Complaints, mentions of returns, regret, bad quality → negative.
- You will be given a list of sentences with indices (0, 1, 2, ...).
- Return a JSON structure with a list called "results".
- Each item in "results" must have:
  - "index": the sentence index from the list
  - "label": "positive" or "negative"

Make sure you cover ALL sentences that are provided.
"""


# ============================
# 4. Helpers
# ============================

def iter_sentences(path: str, max_sentences: int | None = None):
    """Yield sentences (text) from a JSONL file, up to max_sentences."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_sentences is not None and count >= max_sentences:
                break
            obj = json.loads(line)
            # adjust key if your jsonl structure changes
            text = obj.get("text", "").strip()
            if not text:
                continue
            yield text
            count += 1


def batch(iterator, size: int):
    """Yield lists of items of length up to size."""
    current = []
    for item in iterator:
        current.append(item)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


# ============================
# 5. Main Classification Logic
# ============================

def classify_reviews(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_FILE,
    batch_size: int = BATCH_SIZE,
    max_sentences: int | None = MAX_SENTENCES,
):
    # Open output file in write mode.
    # If you want to resume runs, change "w" -> "a" and handle skipping.
    with open(output_path, "w", encoding="utf-8") as out_f:

        sentence_iter = iter_sentences(input_path, max_sentences=max_sentences)

        total_processed = 0

        for chunk_idx, chunk in enumerate(batch(sentence_iter, batch_size)):
            # Build the prompt with indexed sentences
            # Example:
            # 0: sentence...
            # 1: sentence...
            prompt_lines = []
            for i, s in enumerate(chunk):
                prompt_lines.append(f"{i}: {s}")
            user_prompt = "صنّف الجمل التالية إلى إيجابية أو سلبية:\n\n" + "\n".join(prompt_lines)

            # Call the structured LLM
            result: BatchClassification = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            # Map index -> label, then write each as a jsonl line
            for item in result.results:
                idx = item.index
                label = item.label

                # Safety in case the model returns an invalid index
                if 0 <= idx < len(chunk):
                    sentence = chunk[idx]
                    record = {
                        "sentence": sentence,
                        "label": label,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            out_f.flush()  # ensure data is written to disk batch by batch

            total_processed += len(chunk)
            print(f"✓ Processed batch {chunk_idx + 1}, total sentences: {total_processed}")

        print(f"✔ Done! Total classified sentences: {total_processed}")
        print(f"✔ Output saved to: {output_path}")


# ============================
# 6. Run
# ============================

if __name__ == "__main__":
    classify_reviews()
