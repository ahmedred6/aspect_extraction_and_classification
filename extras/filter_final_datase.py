import json
import fasttext

MODEL_PATH = "lid.176.ftz"
model = fasttext.load_model(MODEL_PATH)

INPUT_PATH = "final_dataset.jsonl"
OUTPUT_PATH = "cleaned_final_dataset.jsonl"


def detect_lang(text: str):
    # clean text
    text = text.replace("\n", " ")
    labels, scores = model.predict(text)
    lang = labels[0].replace("__label__", "")
    prob = float(scores[0])
    return lang, prob


def is_english(text: str):
    lang, _ = detect_lang(text)
    return lang == "en"


def filter_english_reviews(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping malformed line:", line)
                continue

            sentence = sample.get("sentence", "")
            if not isinstance(sentence, str):
                continue

            if is_english(sentence):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Done! Total samples: {total}, English kept: {kept}")


if __name__ == "__main__":
    filter_english_reviews()
