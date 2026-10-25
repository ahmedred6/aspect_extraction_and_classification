import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Optional: XGBoost (comment out if not installed)
from xgboost import XGBClassifier

import torch
from transformers import BertTokenizer, BertModel


# ============================================
# 1. Load ABSA JSONL and build aspect-level DF
# ============================================

DATA_PATH = "final_dataset.jsonl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DROP_CONFLICT = True     # option to drop 'conflict' labels
MAX_LENGTH = 128         # BERT max tokens
BATCH_SIZE = 32          # for embedding extraction


def load_jsonl_absa(path: str) -> pd.DataFrame:
    """
    Each line in the JSONL file:
    {
      "id": int,
      "sentence": str,
      "aspect_terms": [
          {"term": "...", "polarity": "...", "from": int, "to": int}
      ]
    }

    Returns a DataFrame with one row per aspect-term:
      - id
      - sentence
      - aspect
      - polarity
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sent = obj["sentence"]
            for asp in obj.get("aspect_terms", []):
                rows.append({
                    "id": obj["id"],
                    "sentence": sent,
                    "aspect": asp["term"],
                    "polarity": asp["polarity"]
                })
    df = pd.DataFrame(rows)
    return df


df = load_jsonl_absa(DATA_PATH)
print("Raw rows:", len(df), " | label counts:\n", df["polarity"].value_counts())

if DROP_CONFLICT and "conflict" in df["polarity"].unique():
    df = df[df["polarity"] != "conflict"].reset_index(drop=True)
    print("\nAfter dropping 'conflict':", len(df))
    print(df["polarity"].value_counts())

sentences = df["sentence"].values
aspects = df["aspect"].values
labels = df["polarity"].values


# ============================================
# 2. Build BERT embeddings (frozen extractor)
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()


def build_bert_embeddings(aspects, sentences, batch_size=32, max_length=128):
    """
    aspects, sentences: arrays or lists of same length
    Returns: np.ndarray [n_samples, hidden_size] (CLS embeddings)
    """
    assert len(aspects) == len(sentences)
    all_vecs = []

    with torch.no_grad():
        for start in range(0, len(aspects), batch_size):
            end = start + batch_size
            batch_aspects = list(aspects[start:end])
            batch_sents = list(sentences[start:end])

            # Use aspect as text A, sentence as text B → BERT sees relation
            enc = tokenizer(
                batch_aspects,
                batch_sents,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # CLS token embedding: [batch_size, hidden_size]
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_vecs.append(cls_embeddings.cpu().numpy())

    X = np.vstack(all_vecs)
    return X


print("\nBuilding BERT embeddings for ALL aspect instances...")
X_bert = build_bert_embeddings(aspects, sentences,
                               batch_size=BATCH_SIZE,
                               max_length=MAX_LENGTH)

print("BERT embedding matrix shape:", X_bert.shape)  # (n_samples, 768)


# ============================================
# 3. Train / Test split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_bert,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])


# ============================================
# 4. Model candidates (classical ML)
# ============================================

models = {
    "logreg": LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight=None,
        random_state=RANDOM_STATE
    ),
    "svm": LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax" if len(np.unique(labels)) > 2 else "binary:logistic",
        eval_metric="mlogloss"
    )
}


# ============================================
# 5. Stratified K-Fold comparison (on TRAIN only)
# ============================================

def evaluate_kfold(model, X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return scores.mean(), scores.std()


print("\n=== STRATIFIED K-FOLD MODEL SELECTION (TRAIN SET ONLY) ===")
cv_results = {}
for name, model in models.items():
    mean_acc, std_acc = evaluate_kfold(model, X_train, y_train, folds=5)
    cv_results[name] = (mean_acc, std_acc)
    print(f"{name}: mean={mean_acc:.4f}  std={std_acc:.4f}")

# Pick best by mean CV accuracy
best_model_name = max(cv_results, key=lambda k: cv_results[k][0])
best_cv_mean, best_cv_std = cv_results[best_model_name]
print(f"\n>>> Best base model: {best_model_name} (CV mean={best_cv_mean:.4f}, std={best_cv_std:.4f})")


# ============================================
# 6. Grid search hyperparameter tuning on BEST model
# ============================================

print("\n=== GRID SEARCH on best model ===")

if best_model_name == "logreg":
    base_estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    param_grid = {
        "C": [0.01, 0.1, 1, 5, 10],
        "penalty": ["l2"],
        "class_weight": [None, "balanced"]
    }

elif best_model_name == "svm":
    base_estimator = LinearSVC(random_state=RANDOM_STATE)
    param_grid = {
        "C": [0.01, 0.1, 1, 5, 10]
    }

elif best_model_name == "xgboost":
    base_estimator = XGBClassifier(
        objective="multi:softmax" if len(np.unique(labels)) > 2 else "binary:logistic",
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

else:
    raise ValueError("Unknown best model.")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    estimator=base_estimator,
    param_grid=param_grid,
    scoring="accuracy",
    cv=skf,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\nBest params from GridSearch:")
print(grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_estimator = grid.best_estimator_


# ============================================
# 7. Final evaluation on TEST SET
# ============================================

print("\n=== FINAL TEST EVALUATION ===")
y_pred = best_estimator.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

