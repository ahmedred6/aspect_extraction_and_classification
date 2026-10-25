# absa/features.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def build_vectorizer():
    """
    Build a FeatureUnion of:
      - word-level TF-IDF (1–2 grams)
      - char-level TF-IDF (3–5 char grams)
    Input: one string per sample (we'll feed "window" column).
    """
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        analyzer="word"
    )

    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=20000,
        sublinear_tf=True,
        analyzer="char"
    )

    vectorizer = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    return vectorizer

