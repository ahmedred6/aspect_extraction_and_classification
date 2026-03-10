import json
import xml.etree.ElementTree as ET

INPUT_FILE = "Laptop_Train_v2.xml"
OUTPUT_FILE = "Laptop_Train_v2.jsonl"

def convert_xml_to_jsonl(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for sentence in root.findall("sentence"):
            sent_id = sentence.get("id")
            text = sentence.find("text").text.strip()

            # Extract valid aspect terms
            aspects = []
            aspect_terms = sentence.find("aspectTerms")
            if aspect_terms is not None:
                for asp in aspect_terms.findall("aspectTerm"):
                    polarity = asp.get("polarity")
                    # Skip unwanted polarities
                    if polarity in ["neutral", "conflict"]:
                        continue

                    aspects.append({
                        "term": asp.get("term"),
                        "polarity": polarity,
                        "from": int(asp.get("from")),
                        "to": int(asp.get("to")),
                    })

            # Build JSON object
            item = {
                "id": int(sent_id),
                "sentence": text,
                "aspect_terms": aspects
            }

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Finished! Saved JSONL to: {output_file}")


if __name__ == "__main__":
    convert_xml_to_jsonl(INPUT_FILE, OUTPUT_FILE)
