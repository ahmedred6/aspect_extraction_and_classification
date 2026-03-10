import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

from features import token_features


FEATURES_FILE = "Hamzah`s_Part/arabic_ate_features.json"
MODEL_PATH     = "Hamzah`s_Part/models/arabic_svm_grid.pkl"
VECTORIZER_PATH = "Hamzah`s_Part/models/arabic_svm_vectorizer.pkl"


def load_dataset():
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
    feats = []
    labels = []

    for entry in dataset:
        tokens = entry["tokens"]
        lbls   = entry["labels"]

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

    print("[INFO] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # ------------------------------
    # 🔥 BEST PARAMETER GRID FOR LinearSVC
    # ------------------------------
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

    print("\n===== BEST PARAMETERS FOUND =====")
    print(grid.best_params_)

    print("\n===== BEST MACRO F1 DURING GRID SEARCH =====")
    print(grid.best_score_)

    # Train final model with best parameters
    best_model = grid.best_estimator_

    print("\n===== EVALUATING ON HELD-OUT TEST SET =====")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)
    print(f"[SAVED] Best model saved → {MODEL_PATH}")
    print(f"[SAVED] Vectorizer saved → {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_with_grid_search()
