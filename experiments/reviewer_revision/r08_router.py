"""Reproduce the Smart Router results (Table 2, Fig. 2) with the dataset received from Maya on 2026-11-23.

Runs src/1_router/train_router.py UNCHANGED; only redirects its outputs (model files, confusion-matrix image)
to a scratch folder so the tracked repo files are not overwritten, and captures the confusion matrix.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import sklearn.metrics as skm

from _common import REPO, load_original, save_results

os.chdir(REPO)
router = load_original("src/1_router/train_router.py", "train_router")
scratch = tempfile.mkdtemp(prefix="r08_router_")
router.MODEL_DIR = scratch
plt.savefig = lambda *a, **k: None          # do not overwrite docs/figures/router_confusion_matrix.png

captured = {}
_cm = skm.confusion_matrix
def cm_spy(*a, **k):
    m = _cm(*a, **k); captured["cm"] = m.tolist(); return m
router.confusion_matrix = cm_spy

router.run_fast_training()
print("confusion matrix [rows = true DIALECT, MSA; cols = predicted]:", captured.get("cm"))
save_results("r08_router", {"confusion_matrix": captured.get("cm"), "scratch_model_dir": scratch},
             paper={"cv_accuracy_LR": 0.8392, "cv_accuracy_ensemble": 0.7328, "cv_accuracy_DT": 0.7271,
                    "training_fit_LR": 0.8858, "fig2": {"dialect_correct": 2068, "msa_correct": 2113,
                                                         "msa_as_dialect": 247, "dialect_as_msa": 292}})
