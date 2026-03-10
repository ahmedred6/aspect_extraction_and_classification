def token_features(tokens, idx):
    token = tokens[idx]
    feats = {
        "token": token,
        "lower": token.lower(),
        "isdigit": token.isdigit(),
        "prefix1": token[:1],
        "prefix2": token[:2],
        "suffix1": token[-1:],
        "suffix2": token[-2:],
    }

    if idx > 0:
        feats["prev"] = tokens[idx - 1].lower()
    else:
        feats["prev"] = "<START>"

    if idx < len(tokens) - 1:
        feats["next"] = tokens[idx + 1].lower()
    else:
        feats["next"] = "<END>"

    return feats
