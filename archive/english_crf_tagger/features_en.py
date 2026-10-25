# features_en.py
import re

def word_shape(token):
    shape = ""
    for c in token:
        if c.isupper():
            shape += "X"
        elif c.islower():
            shape += "x"
        elif c.isdigit():
            shape += "d"
        else:
            shape += c
    return shape


def token_features(tokens, pos_tags, idx):
    token = tokens[idx]
    pos = pos_tags[idx]

    feats = {
        "token": token,
        "lower": token.lower(),
        "is_digit": token.isdigit(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "shape": word_shape(token),
        "prefix1": token[:1],
        "prefix2": token[:2],
        "suffix1": token[-1:],
        "suffix2": token[-2:],
        "pos": pos,
    }

    # Context window (prev 2, next 2)
    for dist in [-2, -1, 1, 2]:
        j = idx + dist
        if 0 <= j < len(tokens):
            feats[f"t{dist}"] = tokens[j].lower()
            feats[f"p{dist}"] = pos_tags[j]
        else:
            feats[f"t{dist}"] = "<OUT>"
            feats[f"p{dist}"] = "<OUT>"

    return feats
