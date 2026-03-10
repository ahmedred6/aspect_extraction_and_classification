# prepare_english_dataset.py
import json
import re
from tqdm import tqdm

INPUT_FILE = "Laptop_Train_v2.jsonl"
OUTPUT_FILE = "english2/english_ate_features.json"


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def prepare_dataset():
    samples = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)

            sentence = item["sentence"]
            tokens = tokenize(sentence)

            # Build character map
            char_map = []
            idx = 0
            for t in tokens:
                char_map.append((idx, idx + len(t)))
                idx += len(t) + 1

            labels = ["O"] * len(tokens)

            for asp in item["aspect_terms"]:
                if asp["polarity"] not in ["positive", "negative"]:
                    continue

                start = asp["from"]
                end = asp["to"]

                inside = False
                for i, (cstart, cend) in enumerate(char_map):
                    if cstart >= start and cend <= end:
                        if not inside:
                            labels[i] = "B-ASP"
                            inside = True
                        else:
                            labels[i] = "I-ASP"

            for tok, lab in zip(tokens, labels):
                samples.append({
                    "token": tok,
                    "label": lab,
                    "sentence": sentence
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_dataset()
