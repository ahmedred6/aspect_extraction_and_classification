import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn Imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imbalanced Learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = r"cleaned_final_dataset.jsonl"
REPORT_FILENAME = "english_aspect_sentiment_counts.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
WINDOW_SIZE = 5
KFOLDS = 5

# ==========================================
# CLEANING + WINDOWING
# ==========================================
def clean_text(text: str) -> str:
    if not text: return ""
    text = text.strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def char_to_token_window(text: str, start_char: int, end_char: int, window_size: int = 5) -> str:
    if not text: return ""
    tokens, starts = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        starts.append(m.start())

    asp_start_idx, asp_end_idx = None, None
    for i, (tok, s) in enumerate(zip(tokens, starts)):
        e = s + len(tok)
        if s <= start_char < e and asp_start_idx is None: asp_start_idx = i
        if s < end_char <= e: asp_end_idx = i
            
    if asp_start_idx is None: return text
    if asp_end_idx is None: asp_end_idx = asp_start_idx

    left = max(0, asp_start_idx - window_size)
    right = min(len(tokens), asp_end_idx + 1 + window_size)
    window_tokens = tokens[left:right]

    rel_start = asp_start_idx - left
    rel_end = asp_end_idx - left

    window_tokens.insert(rel_start, "ASP_START")
    window_tokens.insert(rel_end + 2, "ASP_END")

    return " ".join(window_tokens)

def load_data(path: str) -> pd.DataFrame:
    print(f">>> Loading JSONL: {path}")

    if not os.path.exists(path):
        print("❌ File not found.")
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                raw_text = obj.get("sentence", "")
                
                for asp in obj.get("aspect_terms", []):
                    term = asp.get("term")
                    polarity = asp.get("polarity")
                    start = asp.get("from")
                    end = asp.get("to")

                    if not term or not polarity: continue

                    try:
                        window = char_to_token_window(raw_text, int(start), int(end), WINDOW_SIZE)
                        rows.append({
                            "term": term,
                            "text": clean_text(window), 
                            "label": polarity
                        })
                    except:
                        continue
            except:
                continue

    return pd.DataFrame(rows)

# ==========================================
# REPORT GENERATION
# ==========================================
def save_aspect_report(df, output_folder):
    if df.empty:
        print("[!] No data for report.")
        return

    report = df.groupby(['term', 'label']).size().unstack(fill_value=0)
    report["Total"] = report.sum(axis=1)
    report = report.sort_values("Total", ascending=False)

    path = os.path.join(output_folder, REPORT_FILENAME)
    report.to_csv(path, encoding="utf-8-sig")

    print(f"[+] Report saved to {path}")
    print(report.head(5))

# ==========================================
# VECTORIZER
# ==========================================
def get_advanced_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1,3),
            max_features=20000,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"\w{1,}"
        )),
        ("char", TfidfVectorizer(
            ngram_range=(3,5),
            max_features=30000,
            sublinear_tf=True,
            analyzer="char_wb"
        ))
    ])

# ==========================================
# STACKED MODEL
# ==========================================
def get_stacking_model():
    estimators = [
        ('svm', LinearSVC(C=0.5, loss='squared_hinge', random_state=RANDOM_STATE, class_weight='balanced')),
        ('logreg', LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced')),
        ('sgd', SGDClassifier(loss='modified_huber', alpha=1e-4, class_weight='balanced', random_state=RANDOM_STATE)),
        ('ridge', RidgeClassifier(class_weight='balanced'))
    ]
    final_estimator = LogisticRegression(class_weight="balanced")
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=KFOLDS,
        n_jobs=-1
    )

# ==========================================
# K-FOLD EVALUATION
# ==========================================
def evaluate_models_kfold(X, y, vectorizer):
    models = {
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "LogReg": LogisticRegression(class_weight="balanced", solver="liblinear"),
        "SGD": SGDClassifier(loss='modified_huber', class_weight="balanced"),
        "Ridge": RidgeClassifier(class_weight="balanced"),
        "Stacked": get_stacking_model()
    }

    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("\n======================")
    print("🔍 STRATIFIED K-FOLD COMPARISON")
    print("======================")

    for name, model in models.items():
        fold_acc = []

        print(f"\n>>> {name}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train_fold = vectorizer.transform(X.iloc[train_idx])
            X_val_fold = vectorizer.transform(X.iloc[val_idx])
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            clf = ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", model)
            ])

            clf.fit(X_train_fold, y_train_fold)
            preds = clf.predict(X_val_fold)

            acc = accuracy_score(y_val_fold, preds)
            fold_acc.append(acc)
            print(f"   Fold {fold}: {acc:.4f}")

        results[name] = np.mean(fold_acc)
        print(f"   → Mean CV Accuracy = {results[name]:.4f}")

    return results

# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix_heatmap(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ==========================================
# MAIN
# ==========================================
def main():
    df = load_data(DATA_PATH)
    if df.empty:
        print("Error: No data.")
        return

    output_folder = os.path.dirname(DATA_PATH) or "."
    save_aspect_report(df, output_folder)

    # Remove unwanted labels
    df = df[~df["label"].isin(["neutral", "conflict"])].reset_index(drop=True)

    X = df["text"]
    y = df["label"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Fit vectorizer only on training set
    vectorizer = get_advanced_vectorizer()
    vectorizer.fit(X_train)

    # ===========================
    # K-FOLD MODEL COMPARISON
    # ===========================
    cv_results = evaluate_models_kfold(X_train, y_train, vectorizer)

    print("\n============================")
    print("🏆 BEST MODEL SELECTION")
    print("============================")

    best_model_name = max(cv_results, key=cv_results.get)
    print(f"Best Model = {best_model_name}  (CV Accuracy = {cv_results[best_model_name]:.4f})")

    # Instantiate the chosen model
    if best_model_name == "Stacked":
        best_model = get_stacking_model()
    elif best_model_name == "LinearSVC":
        best_model = LinearSVC(class_weight="balanced")
    elif best_model_name == "LogReg":
        best_model = LogisticRegression(class_weight="balanced", solver="liblinear")
    elif best_model_name == "SGD":
        best_model = SGDClassifier(loss="modified_huber", class_weight="balanced")
    else:
        best_model = RidgeClassifier(class_weight="balanced")

    final_clf = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", best_model)
    ])

    print("\n>>> Training best model on full training set...")
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    final_clf.fit(X_train_vec, y_train)

    print("\n>>> FINAL TEST EVALUATION")
    y_pred = final_clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 TEST ACCURACY = {acc:.4f}")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix_heatmap(y_test, y_pred, sorted(y.unique()))


if __name__ == "__main__":
    main()
