import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from features import token_features
import joblib


FEATURES_FILE = "Hamzah`s_Part/arabic_ate_features_fixed.json"
MODEL_PATH = "Hamzah`s_Part/models/arabic_svm_ate.pkl"
VECTORIZER_PATH = "Hamzah`s_Part/models/arabic_svm_vectorizer.pkl"


def load_dataset():
    """Load dataset grouped by sentences."""
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = {}

    for row in data:
        sent = row["sentence"]
        if sent not in sentences:
            sentences[sent] = {"tokens": [], "labels": []}
        sentences[sent]["tokens"].append(row["token"])
        sentences[sent]["labels"].append(row["label"])

    return list(sentences.values())


def build_features(dataset):
    """Convert sentences to feature dicts and label list."""
    feats = []
    labels = []

    for entry in dataset:
        tokens = entry["tokens"]
        lbls = entry["labels"]

        for i in range(len(tokens)):
            feats.append(token_features(tokens, i))
            labels.append(lbls[i])

    return feats, labels


def evaluate_model_split(test_size=0.20):

    print("[INFO] Loading dataset...")
    dataset = load_dataset()

    feats, labels = build_features(dataset)

    print("[INFO] Vectorizing features...")
    vec = DictVectorizer()
    X = vec.fit_transform(feats)
    y = np.array(labels)

    # Split dataset
    print(f"[INFO] Splitting dataset (test_size = {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("[INFO] Training model...")
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)

    # Print results
    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    # Save model & vectorizer
    print("[INFO] Saving model...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)

    print(f"[SAVED] Model saved to {MODEL_PATH}")
    print(f"[SAVED] Vectorizer saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    evaluate_model_split(test_size=0.20)
