# absa/fasttext_model.py

import os
import numpy as np
import fasttext
import fasttext.util
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from .config import RANDOM_STATE

FASTTEXT_BIN = "cc.en.300.bin"

def load_fasttext_model():
    """
    Load (or download) pretrained FastText English model.
    """
    if not os.path.exists(FASTTEXT_BIN):
        print("Downloading FastText English model (cc.en.300.bin)...")
        fasttext.util.download_model('en', if_exists='ignore')  # creates cc.en.300.bin
    model = fasttext.load_model(FASTTEXT_BIN)
    return model


def build_fasttext_matrix(texts, ft_model):
    """
    texts: Iterable[str] of input strings (we'll use the 'window' column).
    Returns: np.array [n_samples, dim]
    """
    vectors = []
    for t in texts:
        v = ft_model.get_sentence_vector(t)
        vectors.append(v)
    return np.vstack(vectors)


def train_fasttext_svm(X_train, y_train, X_test, y_test):
    """
    X_*: dense vectors
    """
    clf = LinearSVC(C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n===== FastText + SVM =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    return clf, acc
