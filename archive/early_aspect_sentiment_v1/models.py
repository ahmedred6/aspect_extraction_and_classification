# absa/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from .config import RANDOM_STATE

def get_base_models():
    """
    Return individual base models.
    """
    logreg = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight=None,
        random_state=RANDOM_STATE
    )

    svm = LinearSVC(
        C=1.0
    )

    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    dt = DecisionTreeClassifier(
        max_depth=15,
        random_state=RANDOM_STATE
    )

    models = {
        "logreg": logreg,
        "svm": svm,
        "knn": knn,
        "dt": dt,
    }
    return models


def get_ensemble(models_dict):
    """
    Build a soft voting ensemble (where possible).
    For LinearSVC (no predict_proba), we use hard voting.
    """
    estimators = [(name, m) for name, m in models_dict.items()]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="hard"   # soft requires predict_proba; LinearSVC doesn't have it
    )
    return ensemble
