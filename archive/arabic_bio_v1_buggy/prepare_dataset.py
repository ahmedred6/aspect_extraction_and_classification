import json
import re
from tqdm import tqdm
from normalizer import normalize_arabic

INPUT_FILE = "Hamzah`s_Part/Arabic_Hotels_TrD_V2.jsonl"
OUTPUT_FILE = "Hamzah`s_Part/arabic_ate_features.json"


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def find_all_occurrences(text, sub):
    """
    Returns all (start, end) indices of 'sub' inside 'text'.
    Needed because the same aspect term may appear multiple times.
    """
    positions = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(sub)))
        start = idx + 1
    return positions


def prepare_dataset():
    samples = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)

            # Normalize text
            raw_sentence = item["sentence"]
            sentence = normalize_arabic(raw_sentence)
            tokens = tokenize(sentence)

            # Build char map for token boundaries
            char_map = []
            idx = 0
            for t in tokens:
                char_map.append((idx, idx + len(t)))
                idx += len(t) + 1

            labels = ["O"] * len(tokens)

            # -------------------------------
            # Filter aspects by polarity
            # -------------------------------
            filtered_aspects = [
                asp for asp in item["aspects"]
                if asp["polarity"] in ["positive", "negative"]
            ]

            # -------------------------------
            # Correct IOB Tagging
            # After normalization, the aspect positions must be recomputed
            # -------------------------------
            for asp in filtered_aspects:
                aspect_term_raw = asp["aspectTerm"]
                aspect_term = normalize_arabic(aspect_term_raw)

                # Find all occurrences in normalized text
                occurrences = find_all_occurrences(sentence, aspect_term)

                if not occurrences:
                    continue

                for (start, end) in occurrences:
                    inside = False
                    for i, (cstart, cend) in enumerate(char_map):
                        # If token span is fully inside aspect span:
                        if cstart >= start and cend <= end:
                            if not inside:
                                labels[i] = "B-ASP"
                                inside = True
                            else:
                                labels[i] = "I-ASP"

            # Store final samples
            for tok, lab in zip(tokens, labels):
                samples.append({
                    "token": tok,
                    "label": lab,
                    "sentence": sentence
                })

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved cleaned features to: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_dataset()
