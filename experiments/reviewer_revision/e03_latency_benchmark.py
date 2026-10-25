"""Reviewer item: latency — keep the benchmark, specify hardware, state whether LLM calls are included.

Follows the deployed per-review flow of whole_pipeline/ (Table 15 stages):
  Arabic : language detection -> dialect routing -> aspect extraction -> aspect classification
  English: language detection -> aspect extraction -> aspect classification
LLM dialect->MSA normalization is NOT part of these totals; it is timed separately when an API key
(QWEN_API_KEY or OPENAI_API_KEY) is available, otherwise reported as not measured.
Protocol as in the paper: 105 reviews per pipeline, first 5 are warm-up and excluded, single-threaded CPU.

Models are trained here with the paper's configurations (trained weights do not affect inference cost).
ROUTER: dataset.csv is not in the repo, so the router stage uses the exact router architecture
(char_wb TF-IDF 3-5, 10k features + LogisticRegression) fit on Arabic hotel sentences with placeholder
labels. This is a TIMING PROXY ONLY and must be re-timed once the real router data is available.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import json
import platform
import random
import re
import subprocess
import time

import fasttext
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from _common import REPO, load_original, save_results

N_REVIEWS, N_WARMUP, SEED = 100, 5, 42
DATA = os.path.join(REPO, "experiments", "reviewer_revision", "data")

ar_cls = load_original("src/4_aspect_sentiment/aspect_sentiment_arabic.py", "AspClassArabic")
en_cls = load_original("src/4_aspect_sentiment/aspect_sentiment_english.py", "AspClassEnglish")
ar_pipe = load_original("src/6_pipeline/arabic_pipeline.py", "arabic_pipeline")   # normalize, features, spans, window
ar_ate = load_original("src/3_aspect_extraction/arabic/train_svm_ate.py", "train_svm_ate")
en_norm = load_original("src/3_aspect_extraction/english/normalizer_en.py", "normalizer_en")
en_feat = load_original("src/3_aspect_extraction/english/features_en.py", "features_en")
en_ate = load_original("src/3_aspect_extraction/english/train_svm_gridsearch_en.py", "training_with_grid_search_en")


def hardware():
    def ps(cmd):
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()
    return {"cpu": ps("(Get-CimInstance Win32_Processor).Name"),
            "physical_cores": ps("(Get-CimInstance Win32_Processor).NumberOfCores"),
            "logical_processors": ps("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"),
            "ram_gb": ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"),
            "os": ps("(Get-CimInstance Win32_OperatingSystem).Caption"),
            "python": platform.python_version(), "threads": 1, "gpu_used": False}


# ---------------- training (paper configurations) ----------------
def train_models():
    m = {"lid": fasttext.load_model(os.path.join(REPO, "models", "lid.176.ftz"))}

    arabic_sents = [json.loads(l)["sentence"] for l in open(os.path.join(DATA, "Arabic_Hotels_TrD_V2.jsonl"), encoding="utf-8")]
    m["router_vec"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000)
    Xr = m["router_vec"].fit_transform(arabic_sents)
    m["router_le"] = LabelEncoder().fit(["DIALECT", "MSA"])
    yr = np.arange(len(arabic_sents)) % 2  # placeholder labels: timing proxy only
    m["router"] = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xr, yr)

    ar_ate.FEATURES_FILE = os.path.join(DATA, "arabic_ate_features_fixed.json")
    feats, labels = ar_ate.build_features(ar_ate.load_dataset())
    m["ar_ate_vec"] = DictVectorizer()
    m["ar_ate"] = LinearSVC(class_weight="balanced").fit(m["ar_ate_vec"].fit_transform(feats), labels)

    prep = load_original("src/3_aspect_extraction/english/prepare_dataset_en.py", "prepare_dataset_en")
    prep.INPUT_FILE = os.path.join(DATA, "Laptop_Train_v2.jsonl")
    prep.OUTPUT_FILE = os.path.join(DATA, "english_ate_features_latency.json")
    prep.prepare_dataset()
    en_ate.FEATURES_FILE = prep.OUTPUT_FILE
    feats, labels = en_ate.build_features(en_ate.load_dataset())
    m["en_ate_vec"] = DictVectorizer()
    m["en_ate"] = LinearSVC(C=0.01, class_weight="balanced", max_iter=2000).fit(m["en_ate_vec"].fit_transform(feats), labels)

    for key, mod, df in (("ar", ar_cls, ar_cls.load_data(os.path.join(REPO, ar_cls.DATA_PATH))),
                         ("en", en_cls, en_cls.load_data(os.path.join(DATA, "cleaned_final_dataset.jsonl")))):
        df = df[~df["label"].isin(["conflict", "neutral"])]
        vec = mod.get_advanced_vectorizer()
        X = vec.fit_transform(df["text"])
        m[f"{key}_cls_vec"] = vec
        m[f"{key}_cls"] = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=5)),
                                       ("stacking", mod.get_stacking_model())]).fit(X, df["label"])
        m[f"{key}_cls"].named_steps["stacking"].set_params(n_jobs=1)
    return m


# ---------------- pipeline stages ----------------
def detect_language(text, m):
    (_p, label), = m["lid"].f.predict(text.replace("\n", " ") + "\n", 1, 0.0, "strict")
    return label.replace("__label__", "")


def route(text, m):
    return m["router_le"].inverse_transform(m["router"].predict(m["router_vec"].transform([text])))[0]


def extract(text, m, lang):
    norm = ar_pipe.normalize_arabic(text) if lang == "ar" else en_norm.normalize_english(text)
    tokens, spans = ar_pipe.tokenize_with_spans(norm)
    if not tokens:
        return norm, []
    feat_fn = ar_pipe.token_features if lang == "ar" else en_feat.token_features
    labels = m[f"{lang}_ate"].predict(m[f"{lang}_ate_vec"].transform([feat_fn(tokens, i) for i in range(len(tokens))]))
    return norm, ar_pipe.extract_aspects(tokens, labels, spans)


def classify(norm, aspects, m, lang):
    out = []
    for a in aspects:
        window = ar_pipe.clean_text(ar_pipe.char_to_token_window(norm, a["from"], a["to"], window_size=5))
        out.append(m[f"{lang}_cls"].predict(m[f"{lang}_cls_vec"].transform([window]))[0])
    return out


def time_pipeline(sentences, m, lang):
    stages = {"language_detection": [], "aspect_extraction": [], "aspect_classification": []}
    if lang == "ar":
        stages = {"language_detection": [], "dialect_routing": [], "aspect_extraction": [], "aspect_classification": []}
    n_aspects = []
    for s in sentences:
        t = time.perf_counter(); detect_language(s, m); stages["language_detection"].append(time.perf_counter() - t)
        if lang == "ar":
            t = time.perf_counter(); route(s, m); stages["dialect_routing"].append(time.perf_counter() - t)
        t = time.perf_counter(); norm, aspects = extract(s, m, lang); stages["aspect_extraction"].append(time.perf_counter() - t)
        t = time.perf_counter(); classify(norm, aspects, m, lang); stages["aspect_classification"].append(time.perf_counter() - t)
        n_aspects.append(len(aspects))
    stats = {}
    for k, v in stages.items():
        a = np.array(v[N_WARMUP:]) * 1000
        stats[k] = {"mean_ms": a.mean(), "std_ms": a.std(), "p95_ms": float(np.percentile(a, 95))}
    total = np.sum([np.array(v[N_WARMUP:]) for v in stages.values()], axis=0) * 1000
    stats["total"] = {"mean_ms": total.mean(), "std_ms": total.std(), "p95_ms": float(np.percentile(total, 95)),
                      "reviews_per_sec": 1000 / total.mean()}
    stats["mean_aspects_per_review"] = float(np.mean(n_aspects[N_WARMUP:]))
    return stats


def time_llm(sentences):
    """Dialect->MSA LLM call latency (network + generation), only if a key is available."""
    key = os.getenv("QWEN_API_KEY")
    if not key:
        return {"measured": False, "reason": "no QWEN_API_KEY in environment"}
    from langchain_openai import ChatOpenAI
    wp = open(os.path.join(REPO, "src", "7_dashboard", "whole_pipeline.py"), encoding="utf-8").read()
    prompt = re.search(r'MSA_PROMPT = """(.*?)"""', wp, re.S).group(1)
    llm = ChatOpenAI(api_key=key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                     model="qwen-plus-2025-04-28", temperature=0, max_tokens=3000)
    times = []
    for s in sentences[:25]:
        t = time.perf_counter()
        llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": s}])
        times.append(time.perf_counter() - t)
    a = np.array(times[2:]) * 1000
    return {"measured": True, "model": "qwen-plus-2025-04-28", "n": len(a),
            "mean_ms": a.mean(), "std_ms": a.std(), "p95_ms": float(np.percentile(a, 95))}


def main():
    hw = hardware()
    print(hw)
    m = train_models()
    rng = random.Random(SEED)
    ar_s = [json.loads(l)["sentence"] for l in open(os.path.join(DATA, "Arabic_Hotels_TrD_V2.jsonl"), encoding="utf-8")]
    en_s = [json.loads(l)["sentence"] for l in open(os.path.join(REPO, "data", "raw", "amazon_absa_gpt41_labeled.jsonl"), encoding="utf-8")]
    ar_sample = rng.choices(ar_s, k=N_REVIEWS + N_WARMUP)
    en_sample = rng.choices(en_s, k=N_REVIEWS + N_WARMUP)

    res = {"hardware": hw, "protocol": {"reviews_per_pipeline": N_REVIEWS + N_WARMUP, "warmup_excluded": N_WARMUP},
           "llm_included_in_totals": False,
           "router_stage_note": "timing proxy (real router data not available)",
           "arabic": time_pipeline(ar_sample, m, "ar"), "english": time_pipeline(en_sample, m, "en"),
           "llm_dialect_to_msa": time_llm(ar_sample)}
    for lang in ("arabic", "english"):
        print(f"\n{lang}")
        for k, v in res[lang].items():
            print(f"  {k:24s} {v}")
    print("\nLLM:", res["llm_dialect_to_msa"])
    save_results("e03_latency_benchmark", res, paper={
        "arabic_ms": {"language_detection": 0.13, "dialect_routing": 0.79, "aspect_extraction": 0.49,
                      "aspect_classification": 5.02, "total": 6.44},
        "english_ms": {"language_detection": 0.15, "aspect_extraction": 0.82, "aspect_classification": 2.51,
                       "total": 3.48}})


if __name__ == "__main__":
    main()
