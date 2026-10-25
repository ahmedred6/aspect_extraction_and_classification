# absa/data_loader.py

from lxml import etree

def load_semeval_xml(path: str):
    """
    Parse SemEval ABSA XML file into a list of dicts:
    {
      "id": str,
      "text": str,
      "aspects": [
         {"term": str, "polarity": str, "from": int, "to": int},
         ...
      ]
    }
    """
    tree = etree.parse(path)
    root = tree.getroot()

    data = []

    for sentence in root.findall("sentence"):
        sent_id = sentence.get("id")
        text_elem = sentence.find("text")
        if text_elem is None:
            continue
        text = text_elem.text.strip()

        aspects = []
        aspect_terms = sentence.find("aspectTerms")

        if aspect_terms is not None:
            for term in aspect_terms.findall("aspectTerm"):
                aspects.append({
                    "term": term.get("term"),
                    "polarity": term.get("polarity"),
                    "from": int(term.get("from")),
                    "to": int(term.get("to")),
                })

        data.append({
            "id": sent_id,
            "text": text,
            "aspects": aspects,
        })

    return data

