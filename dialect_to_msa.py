import os
import json
from time import sleep
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List


# ===============================
# Load Qwen API Key
# ===============================
load_dotenv(".env")

"""llm_raw = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-04-28",
    temperature=0,
    max_tokens=3000,
)"""

llm_raw = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",      # or "gpt-4o" or "gpt-4o-mini"
    max_retries=3,
    timeout=None
)
# ===============================
# TOOL SCHEMA (bind_tools)
# ===============================

class ConvertedSentence(BaseModel):
    msa: str = Field(description="Sentence rewritten in Modern Standard Arabic (MSA)")

class ConversionResult(BaseModel):
    sentences: List[ConvertedSentence] = Field(
        description="List of rewritten MSA sentences"
    )


# ===============================
# CONFIG
# ===============================
INPUT_FILE = "clean_dialect.jsonl"
OUTPUT_FILE = "clean_msa.jsonl"
BATCH_SIZE = 1


# ===============================
# PROMPT
# ===============================
MSA_PROMPT = """
أنت متخصص في تحويل النصوص العربية المكتوبة باللهجات المختلفة إلى العربية الفصحى الحديثة (MSA).

قواعد التحويل (يجب الالتزام بها بدقة عالية):

1. أعد كتابة الجملة بالعربية الفصحى فقط.
2. يجب الحفاظ على نفس المعنى الأصلي دون أي زيادة أو نقصان.
3. ممنوع منعاً باتاً استبدال الكلمات بمرادفات أو كلمات ذات معنى مختلف.
   مثال: "البرايم" يجب أن تبقى "برايم" أو "خدمة برايم" ولا يجوز تحويلها إلى "الشحن السريع".
4. ممنوع إضافة أو حذف أي معلومة غير مذكورة في النص الأصلي.
5. لا تغيّر الشعور العام ولا نبرة الكاتب.
6. لا تستخدم كلمات عامية.
7. يجب إزالة المدّ الزائد في الحروف، مثل: روووعه → رائعة.
8. يجب الحفاظ على أسماء العلامات التجارية كما هي دون تفسيرها أو استبدالها.
9. أعد كتابة الجمل فقط — دون أي تعليق أو شرح.

سيتم تزويدك بقائمة من الجمل باللهجات المختلفة.
أعد كتابة كل جملة بالعربية الفصحى الحديثة مع الالتزام التام بالقواعد أعلاه.
"""


# ===============================
# Load Reviews
# ===============================
def load_reviews(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append(obj)
    return items


# ===============================
# Convert Batch Using bind_tools
# ===============================
def convert_batch(batch):
    texts = [entry["text"] for entry in batch]

    prompt = MSA_PROMPT + "\n\nالجمل:\n" + json.dumps(texts, ensure_ascii=False, indent=2)

    llm_tool = llm_raw.bind_tools([ConversionResult])

    response = llm_tool.invoke(prompt)

    tool_output = response.tool_calls[0]["args"]

    sentences = tool_output["sentences"]

    return [item["msa"] for item in sentences]


# ===============================
# Save to JSONL
# ===============================
def save_batch(original_batch, msa_results, writer):
    for orig, new_msa in zip(original_batch, msa_results):
        writer.write(json.dumps({
            "asin": orig.get("asin"),
            "page": orig.get("page"),
            "text_original": orig["text"],
            "text_msa": new_msa
        }, ensure_ascii=False) + "\n")


# ===============================
# MAIN
# ===============================
def main():
    reviews = load_reviews(INPUT_FILE)
    total = len(reviews)

    print(f"Loaded {total} raw reviews.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as writer:
        for start in range(0, total, BATCH_SIZE):
            batch = reviews[start:start + BATCH_SIZE]
            batch_num = (start // BATCH_SIZE) + 1

            print(f"Processing batch {batch_num} ...")

            try:
                msa_results = convert_batch(batch)
            except Exception as e:
                print(f"Batch {batch_num} failed:", e)
                continue

            save_batch(batch, msa_results, writer)
            writer.flush()
            sleep(0.3)

    print("DONE. Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
