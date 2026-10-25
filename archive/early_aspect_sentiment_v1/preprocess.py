# absa/preprocess.py

import re

def clean_text(text: str) -> str:
    """
    Simple English text cleaning:
    - lowercasing
    - remove HTML-like tags
    - normalize whitespace
    (You can add more if you want.)
    """
    text = text.strip()
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove tags like <br>
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text
