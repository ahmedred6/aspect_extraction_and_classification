# infer_english_ate.py
import re
import joblib
import spacy
from features_en import token_features

MODEL_PATH = "english2/english_crf_ate.pkl"
nlp = spacy.load("en_core_web_sm")
crf = joblib.load(MODEL_PATH)


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def extract_aspects(sentence):
    tokens = tokenize(sentence)

    doc = nlp(" ".join(tokens))
    pos_tags = [t.tag_ for t in doc]

    feats = [token_features(tokens, pos_tags, i) for i in range(len(tokens))]
    labels = crf.predict_single(feats)

    aspects = []
    i = 0
    idx = 0

    while i < len(tokens):
        tok = tokens[i]
        lab = labels[i]

        # compute character positions
        start_char = idx
        end_char = idx + len(tok)

        if lab == "B-ASP":
            term_tokens = [tok]
            term_start = start_char
            j = i + 1
            idx2 = end_char + 1

            while j < len(tokens) and labels[j] == "I-ASP":
                term_tokens.append(tokens[j])
                idx2 += len(tokens[j]) + 1
                j += 1

            term = " ".join(term_tokens)
            term_end = term_start + len(term)

            aspects.append({
                "term": term,
                "from": term_start,
                "to": term_end
            })

            i = j
            idx = idx2
        else:
            i += 1
            idx = end_char + 1

    return aspects


if __name__ == "__main__":
    text = "The battery life is great but the screen brightness is poor."
    print(extract_aspects(text))
