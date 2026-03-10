INSTRUCTIONS = """
أنت محلل خبير في مهمة استخراج الجوانب وتحليل المشاعر (Aspect-Based Sentiment Analysis)
للمراجعات العربية الخاصة بالمنتجات.

هدفك:
استخراج الجوانب (features/aspects) المذكورة حرفيًا في النص، وتحديد المشاعر تجاه كل جانب.

========================================
📌 أولاً: تعريف "مصطلح الجانب" (Aspect Term)
========================================
مصطلح الجانب يجب أن يكون:
- اسماً أو عبارة اسمية (Noun Phrase)
- تشير إلى جزء من المنتج أو خاصية ملموسة قابلة للتقييم
- مثل: البطارية، الحجم، قوة الشفط، جودة التصنيع، السعر، الضمان، صوت الجهاز، الهيكل.

❌ ممنوع استخراج:
- الأفعال: تنظف، تشفط، تعمل، تنظف بقوة
- الصفات: ممتاز، قوي، جميل، خفيف
- الكلمات العامة: شيء، تجربة، مشكلة (إلا إذا ارتبطت بجزء محدد من المنتج)
- أي كلمة لا تشير إلى ميزة حقيقية للمنتج
- يجب أن يكون مصطلح الجانب مطابقًا للنص الحرفي الموجود في الجملة 100%.
أي اختلاف بسيط (مثل إضافة “الـ”) غير مسموح.

إذا لم يكن في النص أي خاصية ملموسة → يجب إرجاع:
"aspect_terms": []

========================================
📌 ثانياً: قواعد استخراج المشاعر (Polarity)
========================================
يجب تحديد مشاعر كل جانب بناءً على السياق اللغوي الفعلي في النص.

قواعد المشاعر:

1. **positive** عندما يكون هناك:
   - مدح واضح للميزة
   - أداء جيد: قوي، ممتاز، يتتحكم، يوفر، سريع، صامت
   - نتائج إيجابية صراحة

2. **negative** عندما يكون هناك:
   - شكوى أو عدم رضا
   - أوصاف سلبية: صغير جدًا، ضعيف، مزعج، لا يعمل، توقف، صوت عالي
   - عبارات مقارنة ضمنية تشير لسوء الميزة:
     مثل: "صغيرة شوي"، "أقل من المتوقع"

3. **neutral** عندما:
   - يذكر الجانب بدون تقييم واضح
   - لا توجد دلالة قوية على المشاعر
"""

import os
import json
from time import sleep
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
import re


# ===========================
# Schema (Required for bind_tools)
# ===========================

class Aspect(BaseModel):
    term: str = Field(description="Product Feature mentioned literally in the text")
    polarity: Literal["positive", "neutral", "negative"]

class Review(BaseModel):
    id: int = Field(description="Review ID")
    sentence: str = Field(description="The original Arabic review text")
    aspect_terms: List[Aspect]

class Reviews(BaseModel):
    reviews: List[Review] = Field(description="List of extracted review aspect annotations")



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


# ===========================
# Config
# ===========================
INPUT_FILE = "ar_reviews_100k.tsv"
OUTPUT_FILE = "arabic_aspect_results.jsonl"
BATCH_SIZE = 20


# ===========================
# Offset Utility
# ===========================
def similarity(a, b):
    if a.startswith("ال"):
        a = a[2:]
    if b.startswith("ال"):
        b = b[2:]
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max(len(a), len(b))


def add_offsets(sentence, aspects):
    words = sentence.split()

    for asp in aspects:
        original = asp["term"]

        # Step 1: Try literal match
        idx = sentence.find(original)
        if idx != -1:
            asp["from"] = idx
            asp["to"] = idx + len(original)
            continue

        # Step 2: Try stripping "ال"
        if original.startswith("ال"):
            stripped = original[2:]
            idx = sentence.find(stripped)
            if idx != -1:
                asp["term"] = stripped
                asp["from"] = idx
                asp["to"] = idx + len(stripped)
                continue

        # Step 3: Fuzzy match
        best_match = None
        best_score = -1

        for w in words:
            score = similarity(original, w)
            if score > best_score:
                best_match = w
                best_score = score

        if best_score >= 0.6 and best_match:
            idx = sentence.find(best_match)
            asp["term"] = best_match
            asp["from"] = idx
            asp["to"] = idx + len(best_match)
            continue

        # Step 4: Failure
        asp["from"] = -1
        asp["to"] = -1

    return aspects



# ===========================
# Load TSV Reviews
# ===========================
def load_reviews(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header

        for line_num, line in enumerate(f):
            parts = line.strip().split("\t", maxsplit=1)

            if len(parts) < 2:
                continue

            text = parts[1].strip()

            data.append({
                "id": line_num + 1,
                "sentence": text
            })

    return data



# ===========================
# Extract Batch
# ===========================
def extract_batch(batch):
    numbered_json = json.dumps(batch, ensure_ascii=False, indent=2)

    prompt = INSTRUCTIONS + "\n\nالنصوص:\n" + numbered_json

    llm_with_schema = llm.bind_tools([Reviews])

    response = llm_with_schema.invoke(prompt)

    tool_output = response.tool_calls[0]["args"]
    parsed = tool_output["reviews"]

    for entry in parsed:
        entry["aspect_terms"] = add_offsets(entry["sentence"], entry["aspect_terms"])

    return parsed



# ===========================
# Save JSONL
# ===========================
def save_jsonl(results, writer):
    for entry in results:
        writer.write(json.dumps(entry, ensure_ascii=False) + "\n")



# ===========================
# Main Pipeline
# ===========================
def main():
    reviews = load_reviews(INPUT_FILE)
    total = len(reviews)

    print(f"Loaded {total} Arabic reviews.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as writer:
        for start in range(0, total, BATCH_SIZE):
            batch = reviews[start : start + BATCH_SIZE]

            batch_number = (start // BATCH_SIZE) + 1
            print(f"Processing batch {batch_number} ...")

            try:
                results = extract_batch(batch)
            except Exception as e:
                print(f"Batch {batch_number} failed:", e)
                continue

            save_jsonl(results, writer)
            writer.flush()
            sleep(0.3)

    print("DONE. Saved:", OUTPUT_FILE)



if __name__ == "__main__":
    main()
