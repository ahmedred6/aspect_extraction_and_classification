# absa/train.py

import os
import sys
import numpy as np
import pandas as pd
from jsonl_loader import load_jsonl_aspects

from sklearn.model_selection import train_test_split

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    # Add parent dir to path if running directly: python -m absa.train
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absa.config import AMAZON_PATH,DATA_PATH, RANDOM_STATE, TEST_SIZE, WINDOW_SIZE
from absa.data_loader import load_semeval_xml
from absa.aspect_windows import build_apc_dataset_with_windows
from absa.features import build_vectorizer
from absa.models import get_base_models, get_ensemble
from absa.evaluate import evaluate_model, summarize_results
from absa.fasttext_model import load_fasttext_model, build_fasttext_matrix, train_fasttext_svm


def main():
    # ============================
    # 1. Load and build dataset
    # ============================
    print(f"Loading XML data from: {DATA_PATH}")
    parsed_xml = load_semeval_xml(DATA_PATH)
    parsed_jsonl = load_jsonl_aspects(AMAZON_PATH)
    parsed = parsed_jsonl + parsed_xml

    df = build_apc_dataset_with_windows(parsed, window_size=WINDOW_SIZE)
    df = df[df["polarity"]!="conflict"].reset_index(drop=True)
    # Optionally: drop 'conflict' if it’s too rare and hurting training
    # df = df[df["polarity"] != "conflict"].reset_index(drop=True)

    print(f"Total aspect instances: {len(df)}")
    print(df.head())

    # Use aspect-centered window with <ASP> tags as main input
    texts = df["window"].values
    texts_raw = df["sentence_raw"].values
    labels = df["polarity"].values

    # Stratified split for fair evaluation
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    X_train_raw, X_test_raw, _, _ = train_test_split(
        texts_raw, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    # ====================================
    # 2. Classical ML: TF-IDF (word+char)
    # ====================================
    print("\nBuilding TF-IDF (word + char) vectorizer...")
    vectorizer = build_vectorizer()

    print("Fitting TF-IDF on training data...")
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)

    # 2.1 Train individual models
    base_models = get_base_models()
    results = []

    for name, model in base_models.items():
        res = evaluate_model(name, model, X_train_vec, y_train, X_test_vec, y_test)
        results.append(res)

    # 2.2 Train ensemble
    ensemble = get_ensemble(base_models)
    res_ens = evaluate_model("ensemble", ensemble, X_train_vec, y_train, X_test_vec, y_test)
    results.append(res_ens)

    summary_df = summarize_results(results)

    # ====================================
    # 3. FastText + SVM comparison
    # ====================================
    print("\nLoading FastText model...")
    ft_model = load_fasttext_model()

    print("Building FastText sentence vectors for train and test...")
    X_train_ft = build_fasttext_matrix(X_train_raw, ft_model)
    X_test_ft = build_fasttext_matrix(X_test_raw, ft_model)

    ft_clf, ft_acc = train_fasttext_svm(X_train_ft, y_train, X_test_ft, y_test)

    print("\n=== FINAL COMPARISON ===")
    print(summary_df.sort_values("accuracy", ascending=False))
    print(f"\nFastText + SVM accuracy: {ft_acc:.4f}")


if __name__ == "__main__":
    main()


