import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_FILE = "dataset.csv" 
MODEL_DIR    = "t"

def load_data(filepath):
    print(f"   > Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except:
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.lower()
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df.dropna(subset=['label', 'text'], inplace=True)
    df = df[df['label'].isin(['MSA', 'DIALECT'])]
    
    # Quick Balance Check
    msa = df[df['label']=='MSA']
    dia = df[df['label']=='DIALECT']
    min_len = min(len(msa), len(dia))
    
    msa = msa.sample(n=min_len, random_state=42)
    dia = dia.sample(n=min_len, random_state=42)
    df = pd.concat([msa, dia]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def run_fast_training():
    print(">>> 1. PREPARING DATA...")
    df = load_data(DATASET_FILE)
    print(f"   > Total Balanced Data: {len(df)} rows")

    # 2. Vectorize
    print(">>> 2. VECTORIZING...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
    X = vectorizer.fit_transform(df['text'])
    
    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(df['label']) # 0=DIALECT, 1=MSA

    # 3. Define Models (No XGBoost)
    print("\n>>> 3. COMPARING MODELS (Stratified CV)...")
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=20),
        "Ensemble (LR + DT)": VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(class_weight='balanced', max_iter=1000)),
                ('dt', DecisionTreeClassifier(class_weight='balanced', max_depth=20))
            ], voting='soft'
        )
    }

    # 4. The Race
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"   Running {name}...", end="\r")
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        results[name] = scores.mean()
        print(f"   > {name}: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")

    # 5. Pick Winner
    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\n>>> WINNER: {best_name} ({results[best_name]:.2%})")

    # 6. Final Train & Save
    print(f">>> 4. TRAINING FINAL MODEL ({best_name})...")
    best_model.fit(X, y)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(best_model, f"{MODEL_DIR}/router_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/router_vectorizer.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    
    print(f">>> SUCCESS! Saved {best_name} to '{MODEL_DIR}/'")

    # Optional: Quick Confusion Matrix on Training Data
    y_pred = best_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    
    print(f"\nFinal Training Fit Accuracy: {acc:.2%}")
    
    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f"Final Model: {best_name} (Acc: {acc:.1%})")
    plt.savefig("confusion_matrix_final_winner.png")
    print("   > Saved 'confusion_matrix_final_winner.png'")

if __name__ == "__main__":
    run_fast_training()