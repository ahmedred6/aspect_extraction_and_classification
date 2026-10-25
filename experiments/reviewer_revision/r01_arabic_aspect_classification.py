"""Reproduce paper §6.7.3 / §6.7.4 / Fig. 4 — Arabic aspect sentiment classification.

Original script: ROUTING_AND_ASPECT_CLASSIFICATION/AspClassArabic.py
Data:            Arabic_Hotels_Cleaned.xml (SemEval-2016 Task 5, Arabic hotels)

Paper claims: 9,826 samples after dropping neutral/conflict (6,197 pos / 3,629 neg),
stacking accuracy 90.18%, confusion matrix [[626, 100], [93, 1147]].
"""
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from _common import REPO, load_original, save_results

orig = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")


def main():
    df = orig.load_data(os.path.join(REPO, orig.DATA_PATH))
    print(f"Loaded {len(df)} opinion rows")

    report = df.groupby(["term", "label"]).size().unstack(fill_value=0)
    report["Total"] = report.sum(axis=1)
    top5 = report.sort_values("Total", ascending=False).head(5)
    print(top5)

    # --- identical to AspClassArabic.main() from here on ---
    df_model = df[~df["label"].isin(["conflict", "neutral"])].reset_index(drop=True)
    counts = df_model["label"].value_counts().to_dict()
    print("Binary samples:", len(df_model), counts)

    X, y = df_model["text"], df_model["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=orig.TEST_SIZE, stratify=y, random_state=orig.RANDOM_STATE
    )
    vectorizer = orig.get_advanced_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = ImbPipeline([
        ("smote", SMOTE(random_state=orig.RANDOM_STATE, k_neighbors=5)),
        ("stacking", orig.get_stacking_model()),
    ])
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    labels = ["negative", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    neg_recall = cm[0, 0] / cm[0].sum()
    print(f"\nTEST ACCURACY: {acc:.4f}   macro-F1: {macro_f1:.4f}   negative recall: {neg_recall:.4f}")
    print("Confusion matrix [neg, pos]:\n", cm)

    save_results(
        "r01_arabic_aspect_classification",
        {
            "n_opinion_rows": len(df),
            "n_binary": len(df_model),
            "class_counts": counts,
            "n_null_target_rows": int((df["term"] == "NULL").sum()),
            "top5_aspects": top5.reset_index().to_dict(orient="records"),
            "test_accuracy": acc,
            "macro_f1": macro_f1,
            "negative_recall": neg_recall,
            "confusion_matrix": cm.tolist(),
        },
        paper={
            "n_binary": 9826,
            "class_counts": {"positive": 6197, "negative": 3629},
            "test_accuracy": 0.9018,
            "negative_recall": 0.8623,
            "confusion_matrix": [[626, 100], [93, 1147]],
        },
    )


if __name__ == "__main__":
    main()
