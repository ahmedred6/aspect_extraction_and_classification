#load_dataset.py
import json
import pandas as pd
from pathlib import Path

def load_arabic_dataset(path):
    """
    Arabic JSONL format:
    {
        "rid": "...",
        "sentence": "...",
        "aspects": [
            {"aspectTerm": "...", "polarity": "..."}
        ]
    }
    """
    records = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            text = item.get("sentence", "").strip()
            aspects = item.get("aspects", [])

            for asp in aspects:
                records.append({
                    "text": text,
                    "aspect": asp.get("aspectTerm", ""),
                    "polarity": asp.get("polarity", "").lower(),
                    "language": "arabic"
                })

    return pd.DataFrame(records)


def load_english_dataset(path):
    """
    English JSONL format:
    {
        "id": ...,
        "sentence": "...",
        "aspect_terms": [
            {"term": "...", "polarity": "..."}
        ]
    }
    """
    records = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            text = item.get("sentence", "").strip()
            aspects = item.get("aspect_terms", [])

            for asp in aspects:
                records.append({
                    "text": text,
                    "aspect": asp.get("term", ""),
                    "polarity": asp.get("polarity", "").lower(),
                    "language": "english"
                })

    return pd.DataFrame(records)


def load_all_datasets():
    AR_PATH = "Dashboard/gold.jsonl"
    EN_PATH = "Dashboard/cleaned_final_dataset.jsonl"

    df_ar = load_arabic_dataset(AR_PATH)
    df_en = load_english_dataset(EN_PATH)

    df_all = pd.concat([df_ar, df_en], ignore_index=True)

    # Normalize polarity labels (positive/negative/neutral)
    df_all["polarity"] = df_all["polarity"].str.lower()
    df_all["polarity"] = df_all["polarity"].replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral"
    })

    return df_all
