# whole_pipeline.py

import re
import os
import joblib

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

import os

BASE = os.path.dirname(os.path.abspath(__file__))

print(">>> whole_pipeline.py imported successfully BEFORE model loading")

# Import your existing ABSA pipelines
from english_pipeline import analyze_sentence as analyze_english_sentence
from arabic_pipeline import analyze_arabic_sentence
def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # remove tashkeel
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # normalize alef
    text = re.sub(r"[إأآا]", "ا", text)

    # normalize taa marbuta
    text = text.replace("ة", "ه")
    text = re.sub(r'(.)\1+', r'\1', text)

    # normalize ya
    text = re.sub(r"[ىي]", "ي", text)

    return text


# =====================================================
# 1. SIMPLE LANGUAGE DETECTION (NO FASTTEXT)
# =====================================================

# ONE-LINE HEURISTIC: if it has Arabic letters → 'ar', else 'en'
def detect_language(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


# =====================================================
# 2. ROUTER: MSA vs DIALECT (Arabic only)
# =====================================================

ROUTER_MODEL_PATH = os.path.join("routing", "router_model.pkl")
ROUTER_VECTORIZER_PATH = os.path.join("routing", "router_vectorizer.pkl")
ROUTER_LABEL_ENCODER_PATH = os.path.join("routing", "label_encoder.pkl")

router_model = joblib.load(os.path.join(BASE, ROUTER_MODEL_PATH))
router_vec = joblib.load(os.path.join(BASE, ROUTER_VECTORIZER_PATH))
router_le = joblib.load(os.path.join(BASE, ROUTER_LABEL_ENCODER_PATH))  # labels: "MSA", "DIALECT"

def detect_arabic_variant(text: str) -> str:
    """
    Returns 'MSA' or 'DIALECT'
    """
    X = router_vec.transform([text])
    y = router_model.predict(X)[0]
    label = router_le.inverse_transform([y])[0]
    return label


# =====================================================
# 3. LLM FOR DIALECT → MSA CONVERSION
# =====================================================

load_dotenv(".env")

llm_raw = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
     base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-04-28",  
    temperature=0,
    max_tokens=3000,
    max_retries=3,
    timeout=None,
)

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
"""

def convert_dialect_to_msa(text: str) -> str:
    """
    Uses llm_raw to rewrite dialect sentence into MSA.
    """
    messages = [
        {"role": "system", "content": MSA_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = llm_raw.invoke(messages)
    return resp.content.strip()


# =====================================================
# 4. OVERALL POLARITY MODELS (EN + AR)
# =====================================================

# ---- English polarity ----
EN_POLARITY_MODEL_PATH = os.path.join("english_polarity", "english_polarity_model.pkl")
EN_POLARITY_VECTORIZER_PATH = os.path.join("english_polarity", "english_polarity_vectorizer.pkl")

en_polarity_model = joblib.load(os.path.join(BASE, EN_POLARITY_MODEL_PATH))
en_polarity_vec = joblib.load(os.path.join(BASE, EN_POLARITY_VECTORIZER_PATH))


def clean_text_en(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def classify_english_polarity(text: str) -> str:
    clean = clean_text_en(text)
    X = en_polarity_vec.transform([clean])
    pred = en_polarity_model.predict(X)[0]
    return pred


# ---- Arabic polarity ----
AR_POLARITY_MODEL_PATH = os.path.join("arabic_polarity", "arabic_polarity_model.pkl")
AR_POLARITY_VECTORIZER_PATH = os.path.join("arabic_polarity", "arabic_polarity_vectorizer.pkl")

ar_polarity_model = joblib.load(os.path.join(BASE, AR_POLARITY_MODEL_PATH))
ar_polarity_vec = joblib.load(os.path.join(BASE, AR_POLARITY_VECTORIZER_PATH))

def classify_arabic_polarity(text: str) -> str:
    """
    Uses the same normalize_arabic you showed for the polarity training:
        df["clean_text"] = df["text"].apply(normalize_arabic)
    """
    clean = normalize_arabic(text)
    X = ar_polarity_vec.transform([clean])
    pred = ar_polarity_model.predict(X)[0]
    return pred


# =====================================================
# 5. MASTER PIPELINE: analyze_sentence_global
# =====================================================
print(">>> analyze_sentence_global is defined!")
def analyze_sentence_global(text: str):
    """
    Full pipeline:

    1) Detect language (EN vs AR) using Unicode.
    2) If EN:
        - Run english_pipeline.analyze_sentence -> aspects + aspect polarity
        - Run english overall polarity model
    3) If AR:
        - Route (MSA vs DIALECT) with router model
        - If DIALECT: convert to MSA using LLM
        - Run arabic_pipeline.analyze_arabic_sentence on MSA text
        - Run arabic overall polarity model on MSA text
    4) Return unified JSON-style dict.
    """

    lang = detect_language(text)

    # -------------------------
    # English Branch
    # -------------------------
    if lang == "en":
        absa_result = analyze_english_sentence(text)
        overall = classify_english_polarity(text)

        return {
            "input": text,
            "language": "en",
            "variant": "EN",
            "routed_sentence": text,
            "overall_polarity": overall,
            "absa": absa_result,  # already contains aspects + their polarities
        }

    # -------------------------
    # Arabic Branch
    # -------------------------
    # Use original text for router (same as its training)
    variant = detect_arabic_variant(text)  # "MSA" or "DIALECT"

    if variant == "DIALECT":
        msa_text = convert_dialect_to_msa(text)
    else:
        msa_text = text

    # arabic_pipeline handles its own normalization internally
    absa_result = analyze_arabic_sentence(msa_text)
    overall = classify_arabic_polarity(msa_text)

    return {
        "input": text,
        "language": "ar",
        "variant": variant,
        "routed_sentence": msa_text,  # the actual text used for models (MSA)
        "overall_polarity": overall,
        "absa": absa_result,
    }


# =====================================================
# 6. Quick manual test
# =====================================================
if __name__ == "__main__":
    examples = [
        "The battery life is amazing but the keyboard is terrible.",
        "البطارية ممتازة لكن خدمة العملاء سيئة جداً.",
        "الموبايل ده تحفة بس الشبكة زبالة."
    ]

    from pprint import pprint

    for txt in examples:
        print("\n" + "=" * 80)
        print("INPUT:", txt)
        result = analyze_sentence_global(txt)
        pprint(result)
