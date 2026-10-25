# absa/aspect_windows.py

import re
import pandas as pd
from .preprocess import clean_text

def char_to_token_window(text: str, start_char: int, end_char: int, window_size: int = 5) -> str:
    """
    Convert char offsets into a token-level window of ±window_size words
    around the aspect, and insert <ASP> ... </ASP> markers.
    """
    # tokenization by whitespace
    tokens = []
    starts = []
    for m in re.finditer(r'\S+', text):
        tokens.append(m.group())
        starts.append(m.start())

    aspect_start_tok = None
    aspect_end_tok = None

    for i, (tok, s) in enumerate(zip(tokens, starts)):
        e = s + len(tok)
        # aspect start char falls into this token
        if s <= start_char < e and aspect_start_tok is None:
            aspect_start_tok = i
        # aspect end char falls into this (or previous) token
        if s < end_char <= e:
            aspect_end_tok = i
            break

    if aspect_start_tok is None:
        # fallback: full sentence if mapping fails
        return text

    if aspect_end_tok is None:
        aspect_end_tok = aspect_start_tok

    left = max(0, aspect_start_tok - window_size)
    right = min(len(tokens), aspect_end_tok + 1 + window_size)

    window_tokens = tokens[left:right]

    rel_start = aspect_start_tok - left
    rel_end = aspect_end_tok - left

    # Insert tags around aspect span
    window_tokens.insert(rel_start, "<ASP>")
    # +2 because inserting opening tag shifts indices
    window_tokens.insert(rel_end + 2, "</ASP>")

    return " ".join(window_tokens)


def build_apc_dataset_with_windows(parsed_xml, window_size: int = 5) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      - sentence: original sentence text (cleaned)
      - aspect: aspect term (cleaned)
      - polarity: label
      - window: aspect-centered window with <ASP> tags
      - input_full: aspect + [SEP] + full sentence (for comparison)
    """
    rows = []

    for item in parsed_xml:
        raw_text = item["text"]
        text = clean_text(raw_text)

        for asp in item["aspects"]:
            term = asp["term"]
            pol = asp["polarity"]
            start = asp["from"]
            end = asp["to"]

            # Build window using original text (for offsets)
            window_raw = char_to_token_window(raw_text, start, end, window_size=window_size)
            window = clean_text(window_raw)
            aspect_clean = clean_text(term)

            input_full = f"{aspect_clean} [SEP] {text}"

            rows.append({
                "sentence": text,
                "sentence_raw":raw_text,
                "aspect": aspect_clean,
                "polarity": pol,
                "window": window,
                "input_full": input_full,
            })

    df = pd.DataFrame(rows)
    return df

