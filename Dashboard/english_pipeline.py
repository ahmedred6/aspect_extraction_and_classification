# english_pipeline.py (CLEAN + FIXED)

import os
import re
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------
# Normalization
# -----------------------------------------
def normalize_english(text):
    text = str(text)
    text = text.replace("’", "'").replace("“", "\"").replace("”", "\"")
    text = re.sub(r"[^A-Za-z0-9\s,.\-!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------
# Token features
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
        "prev": tokens[idx - 1].lower() if idx > 0 else "<START>",
        "next": tokens[idx + 1].lower() if idx < len(tokens) - 1 else "<END>",
    }
    return feats


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


# -----------------------------------------
# Aspect extraction helper
# -----------------------------------------
def extract_aspects(tokens, labels, text):
    aspects = []
    current_tokens = []
    positions = []
    idx = 0

    char_positions = []
    for token in tokens:
        char_positions.append((idx, idx + len(token)))
        idx += len(token) + 1

    for i, lab in enumerate(labels):
        if lab == "B-ASP":
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": positions[0],
                    "to": positions[-1]
                })
            current_tokens = [tokens[i]]
            positions = [char_positions[i][0], char_positions[i][1]]

        elif lab == "I-ASP" and current_tokens:
            current_tokens.append(tokens[i])
            positions[1] = char_positions[i][1]

    if current_tokens:
        aspects.append({
            "term": " ".join(current_tokens),
            "from": positions[0],
            "to": positions[1]
        })

    return aspects


# -----------------------------------------
# Load models
# -----------------------------------------
extractor_model = joblib.load(
    os.path.join(BASE, "english_aspect_extraction", "english_svm_grid.pkl")
)

extractor_vectorizer = joblib.load(
    os.path.join(BASE, "english_aspect_extraction", "english_svm_vectorizer.pkl")
)

classifier_model = joblib.load(
    os.path.join(BASE, "english_aspect_classification", "english_aspect_classification_model.pkl")
)

classifier_vectorizer = joblib.load(
    os.path.join(BASE, "english_aspect_classification", "english_aspect__classification_vectorizer.pkl")
)


# -----------------------------------------
# Classification window
# -----------------------------------------
def char_to_token_window(text, start, end, window_size=5):
    tokens, starts = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        starts.append(m.start())

    asp_start = asp_end = None
    for i, s in enumerate(starts):
        if s <= start < s + len(tokens[i]):
            asp_start = i
        if s < end <= s + len(tokens[i]):
            asp_end = i

    if asp_start is None:
        return text
    if asp_end is None:
        asp_end = asp_start

    left = max(0, asp_start - window_size)
    right = min(len(tokens), asp_end + 1 + window_size)

    window = tokens[left:right]
    rel_start = asp_start - left
    rel_end = asp_end - left

    window.insert(rel_start, "ASP_START")
    window.insert(rel_end + 2, "ASP_END")

    return " ".join(window)


# -----------------------------------------
# Master English Analyzer
# -----------------------------------------
def analyze_sentence(sentence):
    sent_norm = normalize_english(sentence)
    tokens = tokenize(sent_norm)

    # 1. Build feature vectors for extraction
    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = extractor_vectorizer.transform(feats)

    # 2. Predict IOB tags
    labels = extractor_model.predict(X)

    # 3. Convert IOB → Aspect spans
    aspects = extract_aspects(tokens, labels, sent_norm)

    if not aspects:
        return {"sentence": sentence, "aspects": []}

    # 4. Classify each extracted aspect
    results = []
    for asp in aspects:
        window = char_to_token_window(sent_norm, asp["from"], asp["to"], window_size=5)
        Xw = classifier_vectorizer.transform([window])
        polarity = classifier_model.predict(Xw)[0]

        results.append({
            "term": asp["term"],
            "from": asp["from"],
            "to": asp["to"],
            "polarity": polarity
        })

    return {
        "sentence": sentence,
        "aspects": results
    }
