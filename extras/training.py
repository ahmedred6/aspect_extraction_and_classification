import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def clean_english(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)

def extract_window(text, aspect, window_size=5):
    words = text.split()
    aspect_words = aspect.split()
    n = len(words)
    for i in range(n):
        if words[i:i+len(aspect_words)] == aspect_words:
            start = max(0, i - window_size)
            end = min(n, i + len(aspect_words) + window_size)
            return " ".join(words[start:end])
    return None  


