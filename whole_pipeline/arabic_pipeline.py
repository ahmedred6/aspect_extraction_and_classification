# arabic_full_pipeline.py

import re
import joblib

import re

def normalize_arabic(text):
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

    # Normalize Arabic kaf and ya
    text = text.replace("ي", "ي")
    text = text.replace("ك", "ك")

    # Remove non-Arabic symbols except punctuation
    text = re.sub(r"[^\w\s،؟.]", " ", text, flags=re.UNICODE)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
def token_features(tokens, idx):
    token = tokens[idx]
    feats = {
        "token": token,
        "lower": token.lower(),
        "isdigit": token.isdigit(),
        "prefix1": token[:1],
        "prefix2": token[:2],
        "suffix1": token[-1:],
        "suffix2": token[-2:],
    }

    if idx > 0:
        feats["prev"] = tokens[idx - 1].lower()
    else:
        feats["prev"] = "<START>"

    if idx < len(tokens) - 1:
        feats["next"] = tokens[idx + 1].lower()
    else:
        feats["next"] = "<END>"

    return feats
# ================================
# CONFIG: MODEL PATHS
# ================================
# ATE (Aspect Extraction) model + vectorizer
AR_ABSA_ATE_MODEL_PATH = "whole_pipeline/arabic_aspect_extraction/arabic_svm_grid.pkl"
AR_ABSA_ATE_VECTORIZER_PATH = "whole_pipeline/arabic_aspect_extraction/arabic_svm_vectorizer.pkl"

# Aspect Sentiment Classification model + vectorizer
AR_CLF_MODEL_PATH = "whole_pipeline/arabic_aspect_classification/arabic_aspect_classification_model.pkl"
AR_CLF_VECTORIZER_PATH = "whole_pipeline/arabic_aspect_classification/arabic_aspect_classification_vectorizer.pkl"

WINDOW_SIZE = 5


# ================================
# 1) Shared helpers (same style as classifier)
# ================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def char_to_token_window(text: str, start_char: int, end_char: int, window_size: int = 5) -> str:
    """
    Same logic as in your Arabic classification training code.
    Builds a local window around the aspect span and wraps it with ASP_START / ASP_END.
    """
    if not text:
        return ""

    tokens, starts = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        starts.append(m.start())

    asp_start_idx, asp_end_idx = None, None
    for i, (tok, s) in enumerate(zip(tokens, starts)):
        e = s + len(tok)
        if s <= start_char < e and asp_start_idx is None:
            asp_start_idx = i
        if s < end_char <= e:
            asp_end_idx = i

    if asp_start_idx is None:
        return text
    if asp_end_idx is None:
        asp_end_idx = asp_start_idx

    left = max(0, asp_start_idx - window_size)
    right = min(len(tokens), asp_end_idx + 1 + window_size)
    window_tokens = tokens[left:right]

    # Insert special markers
    rel_start = asp_start_idx - left
    rel_end = asp_end_idx - left
    window_tokens.insert(rel_start, "ASP_START")
    window_tokens.insert(rel_end + 2, "ASP_END")

    return " ".join(window_tokens)


# ================================
# 2) Tokenization with spans (for ATE)
# ================================
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_spans(text):
    """
    Uses the same tokenization pattern as your Arabic ATE training:
      re.findall(r"\\w+|[^\\w\\s]", text, re.UNICODE)
    but via finditer so we also get (start, end) offsets for each token.
    """
    tokens = []
    spans = []  # list of (start, end)
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


# ================================
# 3) Convert IOB tags to aspect spans
# ================================
def extract_aspects(tokens, labels, spans):
    """
    tokens: list of token strings
    labels: list of IOB labels (B-ASP / I-ASP / O)
    spans:  list of (start_char, end_char) for each token (on the SAME text string)
    """
    aspects = []
    current_tokens = []
    start_char = None
    last_end_char = None

    for i, lab in enumerate(labels):
        if lab == "B-ASP":
            # If we already have an ongoing aspect, close it
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": start_char,
                    "to": last_end_char
                })
            # Start a new aspect
            current_tokens = [tokens[i]]
            start_char = spans[i][0]
            last_end_char = spans[i][1]

        elif lab == "I-ASP" and current_tokens:
            current_tokens.append(tokens[i])
            last_end_char = spans[i][1]

        else:
            # label == "O"
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": start_char,
                    "to": last_end_char
                })
                current_tokens = []
                start_char = None
                last_end_char = None

    # Close last aspect if still open
    if current_tokens:
        aspects.append({
            "term": " ".join(current_tokens),
            "from": start_char,
            "to": last_end_char
        })

    return aspects


# ================================
# 4) Load models
# ================================
def load_models():
    # ATE
    ate_model = joblib.load(AR_ABSA_ATE_MODEL_PATH)
    ate_vec = joblib.load(AR_ABSA_ATE_VECTORIZER_PATH)

    # Classification
    clf_model = joblib.load(AR_CLF_MODEL_PATH)
    clf_vec = joblib.load(AR_CLF_VECTORIZER_PATH)

    return ate_model, ate_vec, clf_model, clf_vec


# ================================
# 5) Full Arabic ABSA pipeline for a single sentence
# ================================
def analyze_arabic_sentence(sentence, models=None):
    """
    Runs Arabic Aspect Extraction + Aspect Sentiment Classification
    on a single Arabic sentence.
    Returns a dict:
    {
        "sentence": original_sentence,
        "normalized_sentence": sentence_norm,
        "aspects": [
            {"term": ..., "from": ..., "to": ..., "polarity": ...},
            ...
        ]
    }
    """
    if models is None:
        ate_model, ate_vec, clf_model, clf_vec = load_models()
    else:
        ate_model, ate_vec, clf_model, clf_vec = models

    # 1) Normalize Arabic (same preprocessing used in ATE training)
    sentence_norm = normalize_arabic(sentence)

    # 2) Tokenize normalized sentence with spans
    tokens, spans = tokenize_with_spans(sentence_norm)

    if not tokens:
        return {
            "sentence": sentence,
            "normalized_sentence": sentence_norm,
            "aspects": []
        }

    # 3) Build features for ATE and predict IOB tags
    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = ate_vec.transform(feats)
    labels = ate_model.predict(X)

    # 4) Convert IOB tags to aspect spans (char indices on normalized sentence)
    aspects = extract_aspects(tokens, labels, spans)

    if not aspects:
        return {
            "sentence": sentence,
            "normalized_sentence": sentence_norm,
            "aspects": []
        }

    # 5) For each aspect, build a window and classify sentiment
    results = []
    for asp in aspects:
        term = asp["term"]
        start_char = asp["from"]
        end_char = asp["to"]

        # IMPORTANT: use the SAME text base for indices + window
        # Here we use sentence_norm in both ATE and classification.
        window = char_to_token_window(sentence_norm, start_char, end_char, window_size=WINDOW_SIZE)
        window_clean = clean_text(window)
        Xw = clf_vec.transform([window_clean])
        polarity = clf_model.predict(Xw)[0]

        results.append({
            "term": term,
            "from": start_char,
            "to": end_char,
            "polarity": polarity
        })

    return {
        "sentence": sentence,
        "normalized_sentence": sentence_norm,
        "aspects": results
    }


# ================================
# 6) Quick manual test
# ================================
if __name__ == "__main__":
    # Load models once
    models = load_models()

    # Example Arabic sentence (replace with your own)
    example = "البطارية ممتازة لكن خدمة العملاء سيئة جداً."
    result = analyze_arabic_sentence(example, models=models)

    from pprint import pprint
    pprint(result)
