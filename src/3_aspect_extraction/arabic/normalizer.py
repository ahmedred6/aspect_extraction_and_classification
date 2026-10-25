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
