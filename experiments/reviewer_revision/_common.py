"""Shared helpers for the reproduction harnesses.

Each harness re-runs an ORIGINAL experiment script exactly as the team wrote it
(same functions, same data, same seeds), skipping only plotting windows and
model saving. Results are written to experiments/reviewer_revision/results/<name>.json so they can be
compared against the numbers printed in the paper.
"""
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # original scripts call plt.show(); make it a no-op

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO, "experiments", "reviewer_revision", "results")


def load_original(rel_path, module_name):
    """Import an original script from the repo as a module without running main()."""
    path = os.path.join(REPO, rel_path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_results(name, results, paper=None):
    """Save results (and the paper's reported values, for side-by-side comparison)."""
    import sklearn

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "experiment": name,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "results": results,
        "paper": paper or {},
    }
    out = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[saved] {out}")
    return out
