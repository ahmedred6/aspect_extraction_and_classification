# train_svm_ate_english.py
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

from features_en import token_features


FEATURES_FILE = "english_ate_features.json"
MODEL_PATH = "english/models/english_svm_grid.pkl"
VECTORIZER_PATH = "english/models/english_svm_vectorizer.pkl"


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


def train_with_grid_search():
    print("[INFO] Loading dataset...")
    dataset = load_dataset()
    feats, labels = build_features(dataset)

    vec = DictVectorizer()
    X = vec.fit_transform(feats)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 2, 5],
        "class_weight": [None, "balanced"],
        "loss": ["hinge", "squared_hinge"],
        "max_iter": [2000, 3000, 5000]
    }

    print("[INFO] Running GridSearchCV...")
    grid = GridSearchCV(
        estimator=LinearSVC(),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\n===== BEST PARAMS =====")
    print(grid.best_params_)
    print("\n BEST F1 =", grid.best_score_)

    best_model = grid.best_estimator_

    print("\n===== TEST SET REPORT =====")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)

    print(f"[SAVED] Model → {MODEL_PATH}")
    print(f"[SAVED] Vectorizer → {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_with_grid_search()
