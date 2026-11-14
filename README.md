# Dual-Language Aspect-Based Sentiment Analysis Pipeline

This is the code and data for the paper *Dual-Language Aspect-Based Sentiment Analysis Pipeline*, by Maya Al-Khzaee, Ahmed AbuShawish and Hamzah Alanati.

The system analyzes English, Modern Standard Arabic (MSA) and dialectal Arabic reviews:
1. A **router** separates MSA from dialect, and an **LLM** rewrites dialect into MSA.
2. **SVM BIO taggers** extract aspect terms.
3. **Stacking ensembles** classify the sentiment of each aspect.
4. **General-polarity** classifiers score whole reviews.

The paper's LaTeX source lives in a separate private repo: `ahmedred6/dual-language-absa-paper`.

---

## Working on the paper revision? Start here

The paper is being revised after the reviewers' comments. The LaTeX source, the reviewer-comment tracker and a live review viewer live in a separate private repo, **`ahmedred6/dual-language-absa-paper`** (see its README). This repo holds the evidence behind the revision, in [`experiments/reviewer_revision/`](experiments/reviewer_revision/):

| File | What it gives you |
|---|---|
| [`RESULTS_SUMMARY.md`](experiments/reviewer_revision/RESULTS_SUMMARY.md) | Every number from the reruns and new experiments, plus the **decisions log (D1–D11)** of what the authors agreed to change in the paper |
| [`CODE_FINDINGS.md`](experiments/reviewer_revision/CODE_FINDINGS.md) | What the code actually does (TF-IDF settings, features, models, splits), which answers the Architecture / Methodology / Datasets comments |
| [`LITERATURE_NOTES.md`](experiments/reviewer_revision/LITERATURE_NOTES.md) | Published results of related work, read from the PDFs, with page references |
| `r01`–`r07_*.py` | Reproductions of the paper's original numbers |
| `e01`–`e08_*.py` | New experiments requested by the reviewers |
| `results/*.json` | Saved outputs of every script above |

To rerun an experiment: `python experiments/reviewer_revision/r01_arabic_aspect_classification.py` (any script; outputs go to `results/`). Some scripts need data that must be downloaded first, see `data/README.md`.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Run every script from the repo root.** All paths in the code are relative to it:

```bash
python src/3_aspect_extraction/arabic/xml_to_jsonl.py
python src/3_aspect_extraction/arabic/prepare_bio_dataset.py
python src/3_aspect_extraction/arabic/evaluate_svm_split.py     # → Table 5 (macro-F1 0.7165)
```

Notebooks (`src/5_general_polarity/*.ipynb`) are the one exception: open them in Jupyter from their own folder.

---

## Repository map

```
data/
  raw/                     input datasets (see data/README.md for sources)
  processed/               everything the scripts generate (git-ignored)
models/                    trained models land here, one folder per component (git-ignored .pkl)
  lid.176.ftz              fastText language-ID model (tracked)
src/                       the pipeline, numbered in paper order
  1_router/                MSA vs dialect router                        §6.1
  2_dialect_normalization/ LLM dialect → MSA rewriting                  §3.4
  3_aspect_extraction/     SVM BIO taggers (arabic/, english/)          §6.2, §6.4–6.6
  4_aspect_sentiment/      stacking ensembles per aspect                §6.7
  5_general_polarity/      whole-review sentiment (notebooks)           §6.8
  6_pipeline/              end-to-end pipeline + latency benchmark      §3, §6.9
  7_dashboard/             Streamlit dashboard (prototype)
  data_labeling/           LLM annotation + scraping used to build datasets   §4
experiments/
  reviewer_revision/       reruns and new experiments for the reviewers' comments
                           → start with RESULTS_SUMMARY.md
docs/figures/              figures produced by the code
archive/                   superseded experiments, kept for reference (see archive/README.md)
```

---

## Where does each paper result come from?

| Paper | Script(s), in run order | Data |
|---|---|---|
| **§6.1 Smart Router**, Table 2, Fig. 2 | `src/1_router/train_router.py` | `data/raw/router_dataset.csv` ⚠ not in repo |
| **§3.4 Dialect → MSA** | `src/2_dialect_normalization/dialect_to_msa.py` (needs an LLM API key) | dialect reviews |
| **§6.2 English aspect extraction**, Table 3, Fig. 3 | `src/3_aspect_extraction/english/`: `xml_to_jsonl.py --with-restaurant` → `prepare_dataset_en.py` → `train_svm_gridsearch_en.py` | SemEval-2014 laptops + restaurants |
| **§6.4–6.6 Arabic aspect extraction**, Tables 5–7 | `src/3_aspect_extraction/arabic/`: `xml_to_jsonl.py` → `prepare_bio_dataset.py` → `evaluate_svm_split.py` (T5) / `train_svm_ate.py` (T6) / `ablation_normalization.py` (T7) | SemEval-2016 Arabic hotels |
| **§6.7 Aspect sentiment, Arabic**, Tables 8–9, Fig. 4 | `src/4_aspect_sentiment/aspect_sentiment_arabic.py` | `semeval2016_arabic_hotels_cleaned.xml` |
| **§6.7 Aspect sentiment, English**, Tables 8, 10 | `src/4_aspect_sentiment/filter_english_sentences.py` → `aspect_sentiment_english.py` (5-fold variant: `aspect_sentiment_english_kfold.py`) | `amazon_absa_gpt41_labeled.jsonl` |
| **§6.8 Polarity, English**, Tables 11–12 | `src/5_general_polarity/filter_english_amazon_reviews.py` → `english_polarity.ipynb` | Amazon Review Polarity ⚠ download, see `data/README.md` |
| **§6.8 Polarity, Arabic**, Tables 13–14 | `src/5_general_polarity/arabic_polarity.ipynb` | `arabic_100k_reviews.tsv` |
| **§6.9 Latency**, Table 15 | `src/6_pipeline/benchmark_latency.py` (needs trained models) | |
| **§3 Full pipeline** | `src/6_pipeline/whole_pipeline.py` (needs trained models + LLM key) | |
| **§4 LLM-labelled English ABSA data** | `src/data_labeling/label_english_absa_gpt41.py` → `filter_invalid_offsets.py` | `amazon_reviews_8000.txt` |

Exact, verified reproductions of every number above (with the data rebuilt where needed) are in `experiments/reviewer_revision/` as scripts `r01`–`r07`.

---

## Known gaps

- **Router training data (`router_dataset.csv`, 5,360 LLM-labelled LABR/HARD samples) is not in the repo.** Ask the teammate who built the router.
- **Baselines not in the repo:** the spaCy noun-chunk baseline (§6.2.1) and the CAMeL rule-based baseline (Table 4).
- **Arabic polarity model not saved:** the Arabic polarity notebook never saves its model, but `src/6_pipeline` expects `models/arabic_polarity/arabic_polarity_model.pkl`.
- **No summarization code:** the Qwen guided summarization (§3.5) and the Streamlit dashboard pages are not implemented; `src/7_dashboard` is a prototype.
- **LLM API keys:** the LLM scripts need `OPENAI_API_KEY` / `QWEN_API_KEY` in a `.env` file at the repo root, which is git-ignored.

---

## Old → new locations

For anyone who knew the previous layout (branch `organized`):

| Old | New |
|---|---|
| `ROUTING_AND_ASPECT_CLASSIFICATION/FINALWINNERYAY.py` | `src/1_router/train_router.py` |
| `ROUTING_AND_ASPECT_CLASSIFICATION/AspClassArabic.py` | `src/4_aspect_sentiment/aspect_sentiment_arabic.py` |
| `ROUTING_AND_ASPECT_CLASSIFICATION/AspClassEnglish.py`, `aspClassEnglish2.py` | `src/4_aspect_sentiment/aspect_sentiment_english.py`, `aspect_sentiment_english_kfold.py` |
| ``Hamzah`s_Part/*`` | `src/3_aspect_extraction/arabic/*` (`training_svm_with_gridsearch.py` → `train_svm_gridsearch.py`) |
| `extras/biotagger.py` | `src/3_aspect_extraction/arabic/prepare_bio_dataset.py` |
| `extras/ablation_normalization.py` | `src/3_aspect_extraction/arabic/ablation_normalization.py` |
| `english/*` | `src/3_aspect_extraction/english/*` (`training_with_grid_search_en.py` → `train_svm_gridsearch_en.py`) |
| `polarity/*` | `src/5_general_polarity/*` |
| `whole_pipeline/*`, `extras/benchmark_latency.py` | `src/6_pipeline/*` |
| `Dashboard/*` | `src/7_dashboard/*` |
| `dialect_to_msa.py` | `src/2_dialect_normalization/dialect_to_msa.py` |
| `extras/label*.py`, `test3.py`, `evaluator.py`, scrapers … | `src/data_labeling/*` (renamed by purpose) |
| `extras/filter_final_datase.py` | `src/4_aspect_sentiment/filter_english_sentences.py` |
| `Laptop_Train_v2.xml`, `Arabic_Hotels_Cleaned.xml`, ``Hamzah`s_Part/Arabic_Hotels_TrD_V2.xml``, `ar_reviews_100k.tsv` | `data/raw/` (renamed by source) |
| `final_dataset.jsonl` (branch `main` only) | `data/raw/amazon_absa_gpt41_labeled.jsonl` |
| `lid.176.ftz` | `models/lid.176.ftz` |
| `absa/`, `absa2/`, `english2/`, BERT scripts | `archive/` |
