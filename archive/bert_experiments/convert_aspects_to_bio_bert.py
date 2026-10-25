import json
from transformers import BertTokenizerFast
import pandas as pd
import json
import fasttext

MODEL_PATH = "lid.176.ftz"
model = fasttext.load_model(MODEL_PATH)



def detect_lang(text):
    labels, scores = model.predict(text.replace("\n", " "))
    lang = str(labels[0]).replace("__label__", "")
    prob = float(scores[0])
    return lang, prob


def detect_english(text):
    lang, _ = detect_lang(text)
    return lang == "en"

INPUT_FILE = "aspect_results_cleaned.jsonl"
OUTPUT_FILE = "aspect_extraction/bio_dataset.csv"

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def create_bio_tags(sentence, aspects):
    tokens = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)

    offsets = tokens["offset_mapping"]
    input_ids = tokens["input_ids"]
    token_list = tokenizer.convert_ids_to_tokens(input_ids)

    labels = ["O"] * len(offsets)

    for asp in aspects:
        start = asp["from"]
        end = asp["to"]

        for i, (token_start, token_end) in enumerate(offsets):
            if token_start >= start and token_end <= end:
                if labels[i] == "O":
                    labels[i] = "B-ASP"
                else:
                    labels[i] = "I-ASP"

    return token_list, labels


rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        sentence = obj["sentence"]
        aspects = obj["aspect_terms"]

        toks, labs = create_bio_tags(sentence, aspects)
        if detect_english(sentence):
            rows.append({
            "sentence": sentence,
            "tokens": toks,
            "labels": labs
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("Saved BIO dataset to:", OUTPUT_FILE)

