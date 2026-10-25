# absa/evaluate.py

from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)

    #acc = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {acc:.4f}")
    #print(classification_report(y_test, y_pred, digits=4))
    acc = model.score(X_train,y_train)
    print(f"Accuracy: {acc:.4f}")
    return {
        "model": name,
        "accuracy": acc,
    }


def summarize_results(results_list):
    df = pd.DataFrame(results_list)
    print("\n=== SUMMARY (classical models) ===")
    print(df.sort_values("accuracy", ascending=False))
    return df
