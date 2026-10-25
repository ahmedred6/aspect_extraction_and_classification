import fasttext

MODEL_PATH = "lid.176.ftz"
model = fasttext.load_model(MODEL_PATH)



def detect_lang(text):
    labels, scores = model.predict(text.replace("\n", " "))
    lang = str(labels[0]).replace("__label__", "")
    prob = float(scores[0])
    return lang, prob


def classify(text, threshold=0.60):
    lang, prob = detect_lang(text)

    if lang == "en":
        return "english", lang, prob

    if lang == "ar":
        return "arabic", lang, prob

    return "other", lang, prob