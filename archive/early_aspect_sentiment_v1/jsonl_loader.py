# absa/jsonl_loader.py

import json

def load_jsonl_aspects(path: str):
    """
    Each line in the JSONL file is a dict:
    {
      "id": int,
      "sentence": str,
      "aspect_terms": [
          {"term": "...", "polarity": "...", "from": int, "to": int}
      ]
    }
    Returns the same structure as load_semeval_xml().
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            sentence = obj["sentence"]
            aspects = []

            for asp in obj.get("aspect_terms", []):
                aspects.append({
                    "term": asp["term"],
                    "polarity": asp["polarity"],
                    "from": asp["from"],
                    "to": asp["to"],
                })

            data.append({
                "id": obj["id"],
                "text": sentence,
                "aspects": aspects
            })

    return data
