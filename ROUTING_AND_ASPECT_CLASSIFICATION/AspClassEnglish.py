import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# Sklearn Imports
from sklearn.model_selection import train_test_split
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
# Ensure this matches your actual file path
DATA_PATH = r"cleaned_final_dataset.jsonl"
REPORT_FILENAME = "english_aspect_sentiment_counts.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
WINDOW_SIZE = 5

# ==========================================
# 1. DATA LOADING & CLEANING
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
    
    # Insert special tokens (Model Focus)
    rel_start = asp_start_idx - left
    rel_end = asp_end_idx - left
    window_tokens.insert(rel_start, "ASP_START")
    window_tokens.insert(rel_end + 2, "ASP_END")

    return " ".join(window_tokens)

def load_data(path: str) -> pd.DataFrame:
    print(f">>> Loading JSONL: {path}")
    if not os.path.exists(path):
        # Fallback to create dummy data if file is missing (for testing)
        print("❌ File not found. Please check the path.")
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
                    
                    # Apply Windowing
                    try:
                        window = char_to_token_window(raw_text, int(start), int(end), WINDOW_SIZE)
                        rows.append({
                            "term": term,         # Capture aspect term for reporting
                            "text": clean_text(window), 
                            "label": polarity
                        })
                    except:
                        continue
            except: 
                continue
    
    df = pd.DataFrame(rows)
    return df

# ==========================================
# 2. REPORT GENERATION
# ==========================================
def save_aspect_report(df, output_folder):
    """
    Creates a CSV file counting positives/negatives for each English aspect term.
    """
    print("\n>>> Generating Aspect Sentiment Report...")
    
    if df.empty:
        print("   [!] No data to report.")
        return

    # Group by Term and Label, then count
    report = df.groupby(['term', 'label']).size().unstack(fill_value=0)
    
    # Add a Total column
    report['Total'] = report.sum(axis=1)
    
    # Sort by Total (descending)
    report = report.sort_values(by='Total', ascending=False)
    
    # Save to file
    output_path = os.path.join(output_folder, REPORT_FILENAME)
    report.to_csv(output_path, encoding='utf-8-sig')
    
    print(f"   [+] Report saved to: {output_path}")
    print("   [+] Top 5 Aspects found:")
    print(report.head(5))

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def get_advanced_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=20000, sublinear_tf=True, analyzer="word", token_pattern=r'\w{1,}')),
        ("char", TfidfVectorizer(ngram_range=(3, 5), max_features=30000, sublinear_tf=True, analyzer="char_wb"))
    ])

# ==========================================
# 4. STACKING MODEL
# ==========================================
def get_stacking_model():
    estimators = [
        ('svm', LinearSVC(C=0.5, loss='squared_hinge', random_state=RANDOM_STATE, class_weight='balanced')),
        ('logreg', LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced')),
        ('sgd', SGDClassifier(loss='modified_huber', alpha=1e-4, class_weight='balanced', random_state=RANDOM_STATE)),
        ('ridge', RidgeClassifier(class_weight='balanced'))
    ]
    final_estimator = LogisticRegression(class_weight='balanced')
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1)

# ==========================================
# 5. VISUALIZATION HELPERS
# ==========================================
def plot_confusion_matrix_heatmap(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def print_top_features(vectorizer, model_pipeline, n=20):
    print("\n" + "="*40)
    print(f"TOP {n} MOST INFLUENTIAL FEATURES")
    print("="*40)
    try:
        feature_names = vectorizer.get_feature_names_out()
        stacker = model_pipeline.named_steps['stacking']
        logreg_model = stacker.estimators_[1] 
        coefs = logreg_model.coef_[0]
        top_positive_indices = np.argsort(coefs)[-n:][::-1]
        top_negative_indices = np.argsort(coefs)[:n]

        print(f"\n🟢 TOP POSITIVE INDICATORS (Model predicts 'positive'):")
        for i in top_positive_indices:
            print(f"   {feature_names[i]:<25} ({coefs[i]:.4f})")

        print(f"\n🔴 TOP NEGATIVE INDICATORS (Model predicts 'negative'):")
        for i in top_negative_indices:
            print(f"   {feature_names[i]:<25} ({coefs[i]:.4f})")
    except Exception as e:
        print(f"Could not extract features: {e}")

# ==========================================
# 6. MAIN PIPELINE
# ==========================================
def main():
    # 1. Load Data
    df = load_data(DATA_PATH)
    if df.empty:
        print("❌ Error: No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(df)} samples.")

    # 2. Generate Report (Before Filtering)
    output_folder = os.path.dirname(DATA_PATH)
    # If path is just a filename, use current dir
    if not output_folder: output_folder = "." 
    save_aspect_report(df, output_folder)

    # 3. Filter for Binary Classification
    print("\n>>> Filtering out 'neutral' and 'conflict'...")
    df_model = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    print(f"Training samples remaining: {len(df_model)}")
    print(f"Class Balance: {df_model['label'].value_counts().to_dict()}")

    X = df_model["text"]
    y = df_model["label"]
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # 5. Vectorize
    print("\n>>> Vectorizing...")
    vectorizer = get_advanced_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6. Train
    print(">>> Training Pipeline (SMOTE + Stacking)...")
    model_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('stacking', get_stacking_model())
    ])
    model_pipeline.fit(X_train_vec, y_train)

    # 7. Evaluate
    print("\n>>> Evaluating...")
    y_pred = model_pipeline.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 TEST ACCURACY: {acc:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, digits=4))

    # 8. Visualizations
    unique_labels = sorted(df_model['label'].unique())
    plot_confusion_matrix_heatmap(y_test, y_pred, unique_labels)
    
    # 9. Feature Importance
    print_top_features(vectorizer, model_pipeline, n=20)
    
        # 10. Save Model + Vectorizer
    print("\n>>> Saving Model and Vectorizer...")

    joblib.dump(model_pipeline, "Maya`sDocumentation/english_aspect_classification/english_aspect_classification_model.pkl")
    joblib.dump(vectorizer, "Maya`sDocumentation/english_aspect_classification/english_aspect__classification_vectorizer.pkl")

    print("✔ Model saved as: aspect_sentiment_model.pkl")
    print("✔ Vectorizer saved as: aspect_vectorizer.pkl")

if __name__ == "__main__":
    main()
    