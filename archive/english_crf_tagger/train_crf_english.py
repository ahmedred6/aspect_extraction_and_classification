# train_crf_english.py
import json
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import spacy
from features_en import token_features

FEATURES_FILE = "english2/english_ate_features.json"
MODEL_PATH = "english2/english_crf_ate.pkl"

nlp = spacy.load("en_core_web_sm")


def load_dataset():
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = {}
    for row in data:
        sent = row["sentence"]
        if sent not in grouped:
            grouped[sent] = {"tokens": [], "labels": []}
        grouped[sent]["tokens"].append(row["token"])
        grouped[sent]["labels"].append(row["label"])

    return list(grouped.values())


def build_sequences(dataset):
    X, y = [], []
    for entry in dataset:
        tokens = entry["tokens"]
        labels = entry["labels"]

        doc = nlp(" ".join(tokens))
        pos_tags = [t.tag_ for t in doc]

        feats = [token_features(tokens, pos_tags, i) for i in range(len(tokens))]
        X.append(feats)
        y.append(labels)
    return X, y


def train():
    dataset = load_dataset()
    X, y = build_sequences(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    print(metrics.flat_classification_report(y_test, y_pred))

    import joblib
    joblib.dump(crf, MODEL_PATH)
    print("[SAVED]", MODEL_PATH)


if __name__ == "__main__":
    train()
