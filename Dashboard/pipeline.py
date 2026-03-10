# Dashboard/pipeline.py

import os
import re
import joblib
PKL_PATH = os.path.join(BASE, "whole_pipeline", "english_aspect_extraction", "english_svm_grid.pkl")

# -----------------------------
# 1. LANGUAGE DETECTION
# -----------------------------
try:
    from lingua import Language, LanguageDetectorBuilder  # your preferred detector

    _LANGUAGES = [Language.ENGLISH, Language.ARABIC]
    _detector = (
        LanguageDetectorBuilder.from_languages(*_LANGUAGES)
        .with_low_accuracy_mode()
        .build()
    )
    _LANG_MAP = {
        Language.ENGLISH: "english",
        Language.ARABIC: "arabic",
    }

    def detect_language(text: str) -> str:
        """Return 'english', 'arabic', or 'other'."""
        if not isinstance(text, str) or not text.strip():
            return "other"
        lang = _detector.detect_language_of(text)
        return _LANG_MAP.get(lang, "other")

except Exception:
    # Fallback: simple Unicode heuristic
    _AR_RE = re.compile(r"[\u0600-\u06FF]")

    def detect_language(text: str) -> str:
        text = str(text)
        if _AR_RE.search(text):
            return "arabic"
        return "english"


# -----------------------------
# 2. ARABIC NORMALIZATION (shared)
# -----------------------------
def normalize_arabic(text: str) -> str:
    text = str(text)

    # Remove diacritics
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # Remove tatweel
    text = re.sub(r"ـ", "", text)

    # Normalize alef forms
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # Normalize taa marbuta
    text = text.replace("ة", "ه")

    # Normalize alif maqsura
    text = text.replace("ى", "ي")

    # Normalize Arabic kaf and ya (kept as-is but explicit)
    text = text.replace("ي", "ي")
    text = text.replace("ك", "ك")

    # Remove non-Arabic symbols except basic punctuation
    text = re.sub(r"[^\w\s،؟.]", " ", text, flags=re.UNICODE)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# 3. LOAD ROUTER + POLARITY MODELS
#    (paths follow your friend's structure: whole_pipeline/...)
# -----------------------------
ROUTER_MODEL_PATH = os.path.join("whole_pipeline", "routing", "router_model.pkl")
ROUTER_VEC_PATH = os.path.join("whole_pipeline", "routing", "router_vectorizer.pkl")
ROUTER_LE_PATH = os.path.join("whole_pipeline", "routing", "label_encoder.pkl")

EN_POLARITY_MODEL_PATH = os.path.join(
    "whole_pipeline", "english_polarity", "english_polarity_model.pkl"
)
EN_POLARITY_VEC_PATH = os.path.join(
    "whole_pipeline", "english_polarity", "english_polarity_vectorizer.pkl"
)

AR_POLARITY_MODEL_PATH = os.path.join(
    "whole_pipeline", "arabic_polarity", "arabic_polarity_model.pkl"
)
AR_POLARITY_VEC_PATH = os.path.join(
    "whole_pipeline", "arabic_polarity", "arabic_polarity_vectorizer.pkl"
)

_router_model = _router_vec = _router_le = None
_en_pol_model = _en_pol_vec = None
_ar_pol_model = _ar_pol_vec = None


def _safe_load(path):
    if os.path.exists(path):
        return joblib.load("whole_pipeline/english_aspect_extraction/english_svm_grid.pkl")
    return None


def _lazy_load_models():
    """Load router + polarity models once."""
    global _router_model, _router_vec, _router_le
    global _en_pol_model, _en_pol_vec
    global _ar_pol_model, _ar_pol_vec

    if _router_model is None:
        _router_model = _safe_load(ROUTER_MODEL_PATH)
        _router_vec = _safe_load(ROUTER_VEC_PATH)
        _router_le = _safe_load(ROUTER_LE_PATH)

    if _en_pol_model is None:
        _en_pol_model = _safe_load(EN_POLARITY_MODEL_PATH)
        _en_pol_vec = _safe_load(EN_POLARITY_VEC_PATH)

    if _ar_pol_model is None:
        _ar_pol_model = _safe_load(AR_POLARITY_MODEL_PATH)
        _ar_pol_vec = _safe_load(AR_POLARITY_VEC_PATH)


def detect_arabic_variant(text: str) -> str:
    """
    Returns 'MSA' or 'DIALECT' if router is available,
    otherwise 'MSA' as a safe default.
    """
    _lazy_load_models()
    if not (_router_model and _router_vec and _router_le):
        return "MSA"

    X = _router_vec.transform([text])
    y = _router_model.predict(X)[0]
    label = _router_le.inverse_transform([y])[0]
    return label


# -----------------------------
# 4. POLARITY HELPERS
# -----------------------------
def _clean_en(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def classify_english_polarity(text: str) -> str | None:
    _lazy_load_models()
    if not (_en_pol_model and _en_pol_vec):
        return None
    X = _en_pol_vec.transform([_clean_en(text)])
    return _en_pol_model.predict(X)[0]


def classify_arabic_polarity(text: str) -> str | None:
    _lazy_load_models()
    if not (_ar_pol_model and _ar_pol_vec):
        return None
    clean = normalize_arabic(text)
    X = _ar_pol_vec.transform([clean])
    return _ar_pol_model.predict(X)[0]


# -----------------------------
# 5. OPTIONAL LLM (stub for later)
# -----------------------------

USE_LLM_DIALECT_TO_MSA_DEFAULT = False  # start WITHOUT LLM


def convert_dialect_to_msa(text: str) -> str:
    """
    Placeholder for the Qwen-based LLM conversion.

    For now: returns the original text (no change).
    Later, you can plug in:

        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
        ...

    and call the Qwen model here.
    """
    return text


# -----------------------------
# 6. IMPORT YOUR ABSA PIPELINES
# -----------------------------
from english_pipeline import analyze_sentence as analyze_english_sentence
from arabic_pipeline import analyze_arabic_sentence


# -----------------------------
# 7. MASTER FUNCTION FOR STREAMLIT
# -----------------------------
def analyze_sentence_global_streamlit(text: str, use_llm: bool = USE_LLM_DIALECT_TO_MSA_DEFAULT):
    """
    High-level wrapper used by the Streamlit dashboard.

    Returns a dict like:
    {
        "input": original text,
        "language": "english"/"arabic"/"other",
        "variant": "EN" / "MSA" / "DIALECT",
        "routed_sentence": text actually sent to Arabic models,
        "overall_polarity": "positive"/"negative"/... or None,
        "absa": {
            "sentence": ...,
            "normalized_sentence": ... (for Arabic),
            "aspects": [
                {"term": ..., "from": ..., "to": ..., "polarity": ...},
                ...
            ]
        }
    }
    """
    lang = detect_language(text)

    # ---------------- ENGLISH BRANCH ----------------
    if lang == "english" or lang == "other":
        absa_result = analyze_english_sentence(text)
        overall = classify_english_polarity(text)

        return {
            "input": text,
            "language": "english" if lang == "english" else "other",
            "variant": "EN",
            "routed_sentence": text,
            "overall_polarity": overall,
            "absa": absa_result,
        }

    # ---------------- ARABIC BRANCH ----------------
    variant = detect_arabic_variant(text)  # "MSA" or "DIALECT"

    if variant == "DIALECT" and use_llm:
        routed = convert_dialect_to_msa(text)
    else:
        routed = text

    absa_result = analyze_arabic_sentence(routed)
    overall = classify_arabic_polarity(routed)

    return {
        "input": text,
        "language": "arabic",
        "variant": variant,
        "routed_sentence": routed,
        "overall_polarity": overall,
        "absa": absa_result,
    }
