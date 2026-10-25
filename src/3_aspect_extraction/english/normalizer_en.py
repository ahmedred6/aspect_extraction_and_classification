# normalizer_en.py
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
