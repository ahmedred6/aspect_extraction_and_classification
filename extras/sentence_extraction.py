import xml.etree.ElementTree as ET

# ================================
# CONFIG
# ================================
XML_FILE = "Laptop_Train_v2.xml"
OUTPUT_FILE = "sentences.txt"
MAX_SENTENCES = 100   # <-- set your limit here
# ================================


def extract_sentences(xml_path, max_sentences):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sentences = []
    
    for s in root.findall("sentence"):
        text_node = s.find("text")
        if text_node is not None and text_node.text:
            sentence_text = text_node.text.strip()
            if sentence_text:
                sentences.append(sentence_text)

        if len(sentences) >= max_sentences:
            break

    return sentences


def write_sentences(sentences, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(line + "\n")


def main():
    print(f"Reading XML from: {XML_FILE}")
    sentences = extract_sentences(XML_FILE, MAX_SENTENCES)

    print(f"Extracted {len(sentences)} sentences.")
    print("Writing to:", OUTPUT_FILE)
    write_sentences(sentences, OUTPUT_FILE)

    print("Done!")


if __name__ == "__main__":
    main()
