import joblib
import re

MODEL_DIR = "whole_pipeline/routing"

# 1. Load Model + Vectorizer + Label Encoder
print(">>> Loading model...")
model = joblib.load(f"{MODEL_DIR}/router_model.pkl")
vectorizer = joblib.load(f"{MODEL_DIR}/router_vectorizer.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

print("✔ Model Loaded Successfully!\n")

# -----------------------------
# TEXT CLEANING (MUST MATCH TRAINING)
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_label(text: str):
    text_clean = clean_text(text)
    X = vectorizer.transform([text_clean])
    y_pred = model.predict(X)[0]
    label = label_encoder.inverse_transform([y_pred])[0]
    return label

# -----------------------------
# TEST SET (TEXT + TRUE LABEL)
# -----------------------------
examples = [
    ("هذا نص فصيح يعبر عن اللغة العربية الفصحى.", "MSA"),
    ("شلونك يا خوي؟ هذا لهجة خليجية واضحة.", "DIALECT"),
    ("شو عم تعمل؟ لهجة شامية.", "DIALECT"),
    ("يهدف هذا البحث إلى تحليل العوامل المؤثرة في جودة النصوص العربية.", "MSA"),
    ("تُعدّ اللغة العربية من أكثر اللغات انتشاراً في العالم.", "MSA"),
    ("تعمل المؤسسات على تطوير أنظمة الذكاء الاصطناعي لخدمة المجتمع.", "MSA"),
    ("يُظهر التقرير نتائج إيجابية تشير إلى تحسن الأداء العام.", "MSA"),
    ("يُعتبر الحفاظ على الهوية الثقافية جزءًا مهمًا من بناء المستقبل.", "MSA"),
    ("شلونك اليوم؟ إن شاء الله بخير.", "DIALECT"),
    ("شو عم تعمل هلق؟", "DIALECT"),
    ("إنت لسه ما خلصتش الشغل ولا إيه؟", "DIALECT"),
    ("واش كندير دابا؟", "DIALECT"),
    ("ترى الموضوع مو لهالدرجة، هدّ نفسك.", "DIALECT")
]

# -----------------------------
# RUN TEST + COMPUTE ACCURACY
# -----------------------------
print(">>> Testing Router Model...\n")

correct = 0
total = len(examples)

for text, true_label in examples:
    pred = predict_label(text)
    is_correct = (pred == true_label)
    
    print(f"Text: {text}")
    print(f"Prediction: {pred} | True: {true_label} | {'✔️ Correct' if is_correct else '❌ Wrong'}\n")
    
    if is_correct:
        correct += 1

accuracy = correct / total
print("=====================================")
print(f"Final Accuracy on Test Set: {accuracy:.2%}")
print("=====================================")

