import joblib
import re
from features import token_features

MODEL_PATH = "Hamzah`s_Part/models/arabic_svm_ate.pkl"
VECTORIZER_PATH = "Hamzah`s_Part/models/arabic_svm_vectorizer.pkl"

clf = joblib.load(MODEL_PATH)
vec = joblib.load(VECTORIZER_PATH)

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def extract_aspects(sentence):
    tokens = tokenize(sentence)
    feats = [token_features(tokens, i) for i in range(len(tokens))]
    X = vec.transform(feats)
    preds = clf.predict(X)

    aspects = []
    current = []

    for tok, tag in zip(tokens, preds):
        if tag == "B-ASP":
            if current:
                aspects.append(" ".join(current))
            current = [tok]
        elif tag == "I-ASP" and current:
            current.append(tok)
        else:
            if current:
                aspects.append(" ".join(current))
                current = []

    if current:
        aspects.append(" ".join(current))

    return aspects
