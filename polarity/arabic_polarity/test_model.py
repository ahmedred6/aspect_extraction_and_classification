import joblib
import re

# ============================================================
# 1. Arabic Normalization (same as training version)
# ============================================================

def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # Remove tashkeel (diacritics)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # Normalize Alef variants
    text = re.sub(r"[إأآا]", "ا", text)

    # Normalize taa marbuta → haa
    text = text.replace("ة", "ه")

    # Remove repeated characters (e.g., رااائع → رائع)
    text = re.sub(r'(.)\1+', r'\1', text)

    # Normalize yaa
    text = re.sub(r"[ىي]", "ي", text)

    return text


# ============================================================
# 2. Load Saved Classifier & TF-IDF Vectorizer
# ============================================================

MODEL_PATH = "polarity/arabic/arabic_polarity/arabic_polarity_model.pkl"
VECTORIZER_PATH = "polarity/arabic/arabic_polarity/arabic_polarity_vectorizer.pkl"

print("[+] Loading model and vectorizer...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("[+] Loaded successfully!")


# ============================================================
# 3. Function to Predict Sentiment of Any Arabic Text
# ============================================================

def predict_sentiment(text):
    """
    Takes raw Arabic text -> normalizes -> vectorizes -> predicts label.
    """
    cleaned = normalize_arabic(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    return pred


# ============================================================
# 4. Example Tests
# ============================================================

if __name__ == "__main__":
    samples = [
        "هذا المنتج رائع جدا",
        "سيء جدا ولا انصح به",
        "متوسط الجودة",
        "هذا منتج مذهل"
    ]

    for s in samples:
        prediction = predict_sentiment(s)
        print(f"Text: {s}")
        print("Prediction:", "positive" if prediction == 1 else "negative")
        print("-" * 40)
