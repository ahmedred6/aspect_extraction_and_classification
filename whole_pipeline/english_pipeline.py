import re
import joblib

import re

def normalize_english(text):
    text = str(text)

    # remove weird unicode quotes
    text = text.replace("’", "'").replace("“", "\"").replace("”", "\"")

    # remove non-ASCII except punctuation
    text = re.sub(r"[^A-Za-z0-9\s,.\-!?']", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# features_en.py
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

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def extract_aspects(tokens, labels, sentence):
    aspects = []
    current_tokens = []
    start_char = None

    # Build char positions (needed for classifier)
    char_map = []
    idx = 0
    for t in tokens:
        char_map.append((idx, idx + len(t)))
        idx += len(t) + 1

    for i, lab in enumerate(labels):
        if lab == "B-ASP":
            if current_tokens:
                aspects.append({
                    "term": " ".join(current_tokens),
                    "from": start_char,
                    "to": char_map[i-1][1]
                })
            current_tokens = [tokens[i]]
            start_char = char_map[i][0]

        elif lab == "I-ASP" and current_tokens:
            current_tokens.append(tokens[i])

    # last aspect
    if current_tokens:
        aspects.append({
            "term": " ".join(current_tokens),
            "from": start_char,
            "to": char_map[len(tokens)-1][1]
        })

    return aspects


# Aspect Extraction model
extractor_model = joblib.load("whole_pipeline/english_aspect_extraction/english_svm_grid.pkl")
extractor_vec   = joblib.load("whole_pipeline/english_aspect_extraction/english_svm_vectorizer.pkl")

# Aspect Classification model
clf_model = joblib.load("whole_pipeline/english_aspect_classification/english_aspect_classification_model.pkl")
clf_vec   = joblib.load("whole_pipeline/english_aspect_classification/english_aspect__classification_vectorizer.pkl")


def char_to_token_window(text, start_char, end_char, window_size=5):
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
    rel_start = asp_start_idx - left
    rel_end = asp_end_idx - left

    window_tokens.insert(rel_start, "ASP_START")
    window_tokens.insert(rel_end + 2, "ASP_END")

    return " ".join(window_tokens)

def analyze_sentence(sentence):
    sent_norm = normalize_english(sentence)
    tokens = tokenize(sent_norm)

    # 1. Build feature vectors for extraction
    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = extractor_vec.transform(feats)

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
        Xw = clf_vec.transform([window])
        polarity = clf_model.predict(Xw)[0]

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

example = "The battery life is amazing but the keyboard is terrible."

print(analyze_sentence(example))