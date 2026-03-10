import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,  accuracy_score, f1_score
import joblib
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from features import token_features

FEATURES_FILE = "Hamzah`s_Part/arabic_ate_features_fixed.json"
MODEL_PATH = "Hamzah`s_Part/models/arabic_svm_ate.pkl"
VECTORIZER_PATH = "Hamzah`s_Part/models/arabic_svm_vectorizer.pkl"

def load_dataset():
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = {}
    # Group tokens per sentence
    for row in data:
        sent = row["sentence"]
        if sent not in sentences:
            sentences[sent] = {"tokens": [], "labels": []}
        sentences[sent]["tokens"].append(row["token"])
        sentences[sent]["labels"].append(row["label"])

    return list(sentences.values())

def build_features(dataset):
    feats = []
    labels = []

    for entry in dataset:
        tokens = entry["tokens"]
        lbls = entry["labels"]

        for i in range(len(tokens)):
            feats.append(token_features(tokens, i))
            labels.append(lbls[i])

    return feats, labels


def train_svm_kfold():

    dataset = load_dataset()

    # Extract features and labels
    feats, labels = build_features(dataset)

    vec = DictVectorizer()
    X = vec.fit_transform(feats)
    y = np.array(labels)

    # Stratified KFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LinearSVC(class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        f1_scores.append(f1)
        accuracies.append(acc)

        print(f"Fold {fold} Macro F1: {f1:.4f}")
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print("\n====================================")
    print("FINAL CROSS-VALIDATION RESULTS")
    print("====================================")
    print(f"Mean Macro F1: {np.mean(f1_scores):.4f}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

    # Train final model on full dataset
    final_clf = LinearSVC(class_weight="balanced")
    final_clf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(final_clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)

    print(f"\n[SAVED] Model saved to {MODEL_PATH}")
    print(f"[SAVED] Vectorizer saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_svm_kfold()