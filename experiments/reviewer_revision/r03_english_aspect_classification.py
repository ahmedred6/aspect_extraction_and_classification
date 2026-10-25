"""Reproduce paper §6.7.3 / §6.7.5 / Table 10 — English aspect sentiment classification.

Original chain:
  data/raw/amazon_absa_gpt41_labeled.jsonl   (originally final_dataset.jsonl, only on branch `main`)
    -> src/4_aspect_sentiment/filter_english_sentences.py   (fasttext lid.176 keeps English sentences)
    -> cleaned_final_dataset.jsonl
    -> src/4_aspect_sentiment/aspect_sentiment_english.py

Paper: SMOTE + stacking accuracy 80.0%, class balance 9,175 pos / 9,003 neg,
Table 10 top aspects book/movie/story/price/product.
"""
import json
import os

import fasttext
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from _common import REPO, load_original, save_results

DATA_DIR = os.path.join(REPO, "experiments", "reviewer_revision", "data")
FINAL = os.path.join(REPO, "data", "raw", "amazon_absa_gpt41_labeled.jsonl")
CLEANED = os.path.join(DATA_DIR, "cleaned_final_dataset.jsonl")

orig = load_original("src/4_aspect_sentiment/aspect_sentiment_english.py", "AspClassEnglish")


def build_cleaned():
    os.makedirs(DATA_DIR, exist_ok=True)
    # same logic as src/4_aspect_sentiment/filter_english_sentences.py
    lid = fasttext.load_model(os.path.join(REPO, "models", "lid.176.ftz"))
    total = kept = 0
    with open(FINAL, encoding="utf-8") as fin, open(CLEANED, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sent = sample.get("sentence", "")
            if not isinstance(sent, str):
                continue
            # lid.f.predict == lid.predict without the numpy-2-incompatible wrapper
            (_prob, label), = lid.f.predict(sent.replace("\n", " ") + "\n", 1, 0.0, "strict")
            if label.replace("__label__", "") == "en":
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
    print(f"filter: total={total} english kept={kept}")
    return total, kept


def main():
    total, kept = build_cleaned()
    df = orig.load_data(CLEANED)
    report = df.groupby(["term", "label"]).size().unstack(fill_value=0)
    report["Total"] = report.sum(axis=1)
    top5 = report.sort_values("Total", ascending=False).head(5)
    print(top5)

    # --- identical to AspClassEnglish.main() from here on ---
    df_model = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    counts = df_model["label"].value_counts().to_dict()
    print("Binary samples:", len(df_model), counts)
    X, y = df_model["text"], df_model["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=orig.TEST_SIZE, stratify=y, random_state=orig.RANDOM_STATE)
    vectorizer = orig.get_advanced_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = ImbPipeline([
        ("smote", SMOTE(random_state=orig.RANDOM_STATE, k_neighbors=5)),
        ("stacking", orig.get_stacking_model()),
    ])
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])
    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTEST ACCURACY: {acc:.4f}  macro-F1: {mf1:.4f}\n{cm}")

    save_results(
        "r03_english_aspect_classification",
        {"filter_total": total, "filter_kept": kept, "n_binary": len(df_model),
         "class_counts": counts, "top5_aspects": top5.reset_index().to_dict(orient="records"),
         "test_accuracy": acc, "macro_f1": mf1, "confusion_matrix": cm.tolist()},
        paper={"test_accuracy": 0.800, "class_counts": {"positive": 9175, "negative": 9003},
               "cv5_mean_stacking": 0.7922,
               "table10": {"book": [326, 56, 410, 792], "movie": [121, 16, 94, 231],
                           "story": [86, 9, 104, 199], "price": [45, 7, 144, 198],
                           "product": [88, 6, 63, 157]}},
    )


if __name__ == "__main__":
    main()
