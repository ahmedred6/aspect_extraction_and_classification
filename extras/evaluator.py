import json
import xml.etree.ElementTree as ET

# ======================================
# CONFIG
# ======================================
XML_FILE = "Laptop_Train_v2.xml"
PRED_FILE = "aspect_results.jsonl"
SHOW_MAX = 50   # max number of mismatches to print
# ======================================


def load_gold_aspects_by_index(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gold_list = []
    texts = []  # also store original sentence text for debugging

    for s in root.findall("sentence"):
        text_node = s.find("text")
        text_value = text_node.text.strip() if text_node is not None else ""

        aspect_terms = []
        aspect_node = s.find("aspectTerms")

        if aspect_node is not None:
            for term in aspect_node.findall("aspectTerm"):
                aspect_terms.append({
                    "term": term.get("term").strip(),
                    "polarity": term.get("polarity").strip()
                })

        texts.append(text_value)
        gold_list.append(aspect_terms)

    return gold_list, texts


def load_pred_aspects_by_index(pred_path):
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            preds.append(entry.get("aspect_terms", []))
    return preds


def normalize_term(t):
    return t.lower().strip()


def evaluate_and_collect_errors(gold_list, pred_list, texts):
    tp = fp = fn = 0
    mismatches = []

    n = min(len(gold_list), len(pred_list))

    for i in range(n):
        gold_terms = gold_list[i]
        pred_terms = pred_list[i]

        gold_set = {(normalize_term(g["term"]), g["polarity"]) for g in gold_terms}
        pred_set = {(normalize_term(p["term"]), p["polarity"]) for p in pred_terms}

        # Score
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        if gold_set != pred_set:
            mismatches.append({
                "index": i + 1,
                "sentence": texts[i],
                "gold": list(gold_set),
                "pred": list(pred_set),
                "missing": list(gold_set - pred_set),
                "extra": list(pred_set - gold_set)
            })

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1, tp, fp, fn, mismatches


def main():
    gold_list, texts = load_gold_aspects_by_index(XML_FILE)
    pred_list = load_pred_aspects_by_index(PRED_FILE)

    precision, recall, f1, tp, fp, fn, mismatches = evaluate_and_collect_errors(gold_list, pred_list, texts)

    print("\n===== ABSA Evaluation (Term + Polarity, index-aligned) =====")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("============================================\n")

    # Print mismatches
    print(f"=== Showing up to {SHOW_MAX} mismatches ({len(mismatches)} total) ===\n")
    for m in mismatches[:SHOW_MAX]:
        print(f"Sentence #{m['index']}: {m['sentence']}")
        print(f"  GOLD: {m['gold']}")
        print(f"  PRED: {m['pred']}")
        print(f"  Missing aspects: {m['missing']}")
        print(f"  Extra aspects:   {m['extra']}")
        print("---------------------------------------------------------")

    if len(mismatches) > SHOW_MAX:
        print(f"\n... {len(mismatches) - SHOW_MAX} more mismatches not printed.\n")


if __name__ == "__main__":
    main()
