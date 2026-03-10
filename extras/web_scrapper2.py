import json
import fasttext

MODEL_PATH = "lid.176.ftz"
model = fasttext.load_model(MODEL_PATH)



def detect_lang(text):
    labels, scores = model.predict(text.replace("\n", " "))
    lang = str(labels[0]).replace("__label__", "")
    prob = float(scores[0])
    return lang, prob


def classify(text, threshold=0.60):
    lang, prob = detect_lang(text)

    if lang == "en":
        return "english", lang, prob

    if lang == "ar":
        return "arabic", lang, prob

    return "other", lang, prob


# -----------------------------------------------------------
# Run tests on dataset & save results
# -----------------------------------------------------------
INPUT_FILE = "amazon_reviews_arabic.jsonl"
OUTPUT_FILE = "language_classification_results.jsonl"

print("\n🧪 Running FastText Language Detection…")
print(f"Reading: {INPUT_FILE}")
print(f"Writing results to: {OUTPUT_FILE}\n")

# Clear output file
open(OUTPUT_FILE, "w", encoding="utf-8").close()

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

    for line in infile:
        try:
            obj = json.loads(line.strip())
            text = obj["text"]
        except:
            continue

        label, raw, prob = classify(text)

        # Print to terminal
        print(f"[{label}] {text}")
        print(f"    ↳ raw: {raw}, prob: {prob:.4f}\n")

        # Write to output file
        result = {
            "text": text,
            "detected": label,
            "fasttext_raw": raw,
            "prob": prob
        }
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

print("\n✅ DONE! Results recorded in:", OUTPUT_FILE)
