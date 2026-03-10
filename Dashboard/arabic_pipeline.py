# arabic_pipeline.py  (CLEAN + FIXED)

import os
import re
import joblib

# Base directory for model paths
BASE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------
# 1. Normalization
# -----------------------------------------
def normalize_arabic(text):
    text = str(text)

    # Remove diacritics
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # Remove tatweel
    text = re.sub(r"ـ", "", text)

    # Normalize alef forms
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # Normalize taa marbuta
    text = text.replace("ة", "ه")

    # Normalize alif maqsura
    text = text.replace("ى", "ي")

    # Remove odd symbols
    text = re.sub(r"[^\w\s،؟.]", " ", text, flags=re.UNICODE)

    # Collapse spaces
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------
# 2. Token Features
# -----------------------------------------
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

    feats["prev"] = tokens[idx - 1].lower() if idx > 0 else "<START>"
    feats["next"] = tokens[idx + 1].lower() if idx < len(tokens) - 1 else "<END>"

    return feats


# -----------------------------------------
# 3. Load Models (SAFE)
# -----------------------------------------
def load_models():
    ate_model = joblib.load(
        os.path.join(BASE, "arabic_aspect_extraction", "arabic_svm_grid.pkl")
    )
    ate_vec = joblib.load(
        os.path.join(BASE, "arabic_aspect_extraction", "arabic_svm_vectorizer.pkl")
    )
    clf_model = joblib.load(
        os.path.join(BASE, "arabic_aspect_classification", "arabic_aspect_classification_model.pkl")
    )
    clf_vec = joblib.load(
        os.path.join(BASE, "arabic_aspect_classification", "arabic_aspect_classification_vectorizer.pkl")
    )

    return ate_model, ate_vec, clf_model, clf_vec


# Preload once for speed (Streamlit recommended)
AR_ABSA_MODELS = load_models()

WINDOW_SIZE = 5
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# -----------------------------------------
# 4. Tokenization with spans
# -----------------------------------------
def tokenize_with_spans(text):
    tokens, spans = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


# -----------------------------------------
# 5. IOB → Aspect spans
# -----------------------------------------
def extract_aspects(tokens, labels, spans):
    aspects = []
    current_tokens = []
    start_char, last_end = None, None

    for i, lab in enumerate(labels):
        if lab == "B-ASP":
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": start_char,
                    "to": last_end
                })
            current_tokens = [tokens[i]]
            start_char = spans[i][0]
            last_end = spans[i][1]

        elif lab == "I-ASP" and current_tokens:
            current_tokens.append(tokens[i])
            last_end = spans[i][1]

        else:
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": start_char,
                    "to": last_end
                })
                current_tokens = []
                start_char = None

    if current_tokens:
        aspects.append({
            "term": " ".join(current_tokens),
            "from": start_char,
            "to": last_end
        })

    return aspects


# -----------------------------------------
# 6. Build classification window
# -----------------------------------------
def char_to_token_window(text, start_char, end_char, window_size=5):
    tokens, starts = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        starts.append(m.start())

    asp_start, asp_end = None, None
    for i, (tok, s) in enumerate(zip(tokens, starts)):
        if s <= start_char < s + len(tok):
            asp_start = i
        if s < end_char <= s + len(tok):
            asp_end = i

    if asp_start is None:
        return text
    if asp_end is None:
        asp_end = asp_start

    left = max(0, asp_start - window_size)
    right = min(len(tokens), asp_end + 1 + window_size)

    window_tokens = tokens[left:right]
    rel_start = asp_start - left
    rel_end = asp_end - left

    window_tokens.insert(rel_start, "ASP_START")
    window_tokens.insert(rel_end + 2, "ASP_END")

    return " ".join(window_tokens)


# -----------------------------------------
# 7. Full Arabic Sentence Analysis
# -----------------------------------------
def analyze_arabic_sentence(sentence, models=AR_ABSA_MODELS):

    ate_model, ate_vec, clf_model, clf_vec = models

    norm = normalize_arabic(sentence)
    tokens, spans = tokenize_with_spans(norm)

    if not tokens:
        return {"sentence": sentence, "normalized_sentence": norm, "aspects": []}

    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = ate_vec.transform(feats)
    labels = ate_model.predict(X)

    aspects = extract_aspects(tokens, labels, spans)
    results = []

    for asp in aspects:
        window = char_to_token_window(norm, asp["from"], asp["to"])
        window_clean = window.strip().lower()
        polarity = clf_model.predict(clf_vec.transform([window_clean]))[0]

        results.append({
            "term": asp["term"],
            "from": asp["from"],
            "to": asp["to"],
            "polarity": polarity
        })

    return {
        "sentence": sentence,
        "normalized_sentence": norm,
        "aspects": results
    }
