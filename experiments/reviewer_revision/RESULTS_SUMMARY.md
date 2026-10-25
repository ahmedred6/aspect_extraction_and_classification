# Reviewer revision: experiment results

Last updated: 2026-10-25. The paper's facts and numbers stay as they are. This file covers only the reviewer comments that need (re-)experimentation.
Every number here was produced by a script in this folder (`experiments/reviewer_revision/`). The script's output is in `results/<script>.json`. Run the scripts from this folder.
Environment: Python 3.10.11, scikit-learn 1.6.1, imbalanced-learn 0.13.0. Seed 42 throughout.

## Decisions log (apply to the paper later)

| # | Date | Decision | Where in paper |
|---|---|---|---|
| D1 | 2026-10-25 | **Smart Router: no rerun.** Answer the reviewer by clarification. The paper itself describes 88.6% as the fit after retraining on the full dataset: Table 2 "Training Fit 88.58%"; §6.1 and §6.1.2 "retraining on the whole dataset"; Fig. 2 sums to the full 4,720 balanced samples. The held-out estimate is the **83.92%** 5-fold CV accuracy. No number changes. ⚠ **Confirm with the teammate who ran it:** earlier it was stated that 88.6% is a test accuracy. If so, also state the test split size and method | Table 2, §6.1, §6.1.2 |
| D2 | 2026-10-25 | **Replace the English Decision Tree accuracy in Table 8**: 76.5% → **64.8%**, with macro-F1 **0.648** (e05: default DT under the §6.7.1-6.7.2 setup). No claim changes (DT < SVM < Stacking still holds) | Table 8 |
| D3 | 2026-10-25 | **Add macro-F1 to Table 8** for all rows (values in §2 below) | Table 8 |
| D4 | 2026-10-25 | **Add macro-F1 to the polarity tables** (values in §3); note that macro-F1 equals accuracy because the classes are balanced | Tables 11-14 |
| D5 | 2026-10-25 | **Remove the BERT/DistilBERT latency comparison.** The system does not use BERT, and the figures were not measured on the same hardware. Delete the sentences "For context, published CPU inference benchmarks report approximately 25-40 ms per sentence for BERT-base [14] and 15-25 ms for DistilBERT [17] under comparable single-threaded conditions. Both proposed pipelines thus operate at roughly 4-7 times lower latency than a distilled transformer…". Keep the BERT mentions in Related Work and future work: they are background, not claims of use | §6.9 |
| D6 | 2026-10-25 | **Latency:** state the hardware and that the LLM call is excluded. Table 15 hardware (from the machine that ran it): **AMD Ryzen 7 4700U** (8 cores / 8 threads, 2.0 GHz base), **15.4 GB RAM**, **Windows 11 Home** (10.0.26200), CPU only | §6.9 |
| D7 | 2026-10-25 | **English aspect-extraction F1 0.8066 is token-level** (macro over B-ASP/I-ASP/O). State this; optionally add exact-span F1 0.683 | §6.2 |
| D8 | 2026-10-25 | **SMOTE wording.** Keep all reported (with-SMOTE) numbers. Replace the claim that negative recall would "likely collapse" without SMOTE with the measured comparison. Proposed text: *"To assess the contribution of SMOTE, the Arabic stacking ensemble was trained with and without oversampling on the same split. Overall performance was statistically indistinguishable (accuracy 90.13% vs 90.28%, macro-F1 0.894 vs 0.898; McNemar p = 0.83), because all base learners already apply class-balanced weighting. SMOTE instead shifts the balance of errors between classes (negative recall 86.4% vs 92.8%, positive recall 92.3% vs 88.8%). For the English data, which is naturally balanced, SMOTE had no effect."* (e01, e07) | §6.7.2 (and the general claim in §5.3) |

## Status

| # | Reviewer comment | Status | Script |
|---|---|---|---|
| 1 | Aspect sentiment: results with vs without SMOTE | ✅ done (+ significance test; wording D8) | e01, e07 |
| 2 | Aspect sentiment: macro-F1 next to accuracy (Table 8) | ✅ done | e01, e05, e05b |
| 3 | Polarity: macro-F1 next to accuracy (Tables 11-14) | ✅ done | e02, e06 |
| 4 | Leakage-safe rerun (TF-IDF / SMOTE fit only on training data) | ✅ done (Arabic polarity). The router needs no rerun (see #5) | e02 |
| 5 | Smart Router: clarify whether 88.6% is test accuracy | ✅ answered by clarification, no rerun (§5, D1; teammate to confirm) | — |
| 6 | English aspect extraction: token-level vs exact-span F1 | ✅ done: **token-level** (see §6) | r06, r07 |
| 7 | Latency: specify hardware, whether LLM calls are included, drop BERT comparison unless measured | ✅ no experiment needed (see §7) | e03 (optional sanity rerun) |
| 8 | Explain Arabic dataset use (routing vs ABSA) and ensure no test data leaks into training | ✅ answered from code (see §8) | — |

## 1. SMOTE vs no SMOTE (aspect sentiment, Stacking, same 80/20 test set as the paper)

| | Accuracy | Macro-F1 | Negative recall |
|---|---|---|---|
| Arabic, with SMOTE (paper setting) | 0.9013 | 0.8940 | 0.8636 |
| Arabic, without SMOTE | 0.9028 | 0.8980 | 0.9284 |
| English, with SMOTE | 0.8017 | 0.8017 | 0.8060 |
| English, without SMOTE | 0.8017 | 0.8017 | 0.8060 |

- **Arabic:** removing SMOTE does not reduce negative recall. All base learners already use `class_weight="balanced"`.
- **English:** the classes are nearly balanced (7,340 / 7,202 in train), so SMOTE adds about 140 samples and changes nothing.
- **Decide on wording:** §6.7.2 says negative recall would "likely" collapse without SMOTE. The measured result points the other way, and the authors need to decide how to word this.

**Significance (e07, McNemar exact test, Arabic, same 1,966 test items):**

| Comparison | Only with-SMOTE correct | Only without-SMOTE correct | p-value |
|---|---|---|---|
| Overall | 44 | 47 | **0.83 (not significant)** |
| Negative class | 0 | 47 | < 0.001 |
| Positive class | 44 | 0 | < 0.001 |

- **Overall:** SMOTE makes no significant difference.
- **By class:** it significantly **shifts errors between the classes**:
  - negative recall: 0.864 with SMOTE vs 0.928 without;
  - positive recall: 0.923 with SMOTE vs 0.888 without.
- **English:** predictions are identical with and without SMOTE, so there is nothing to test.
- **Impact on the paper:** none of the reported numbers change (all are with SMOTE). Only the explanation in §6.7.2 ("the impact was substantial … sharp collapse in negative recall") and the general claim in §5.3 are affected.

## 2. Table 8 with macro-F1

| Model | English acc / macro-F1 | Arabic acc / macro-F1 |
|---|---|---|
| Decision Tree | 64.8% / 0.648 (replaces 76.5%, see note) | 71.2% / 0.711 |
| Linear SVM | 78.2% / 0.783 | 85.5% / 0.852 |
| Stacking Ensemble | 80.0% / 0.802 | 90.2% / 0.894 |

- **Stacking:** the original runs were reproduced (r01, r03), and macro-F1 comes from the same runs.
- **DT / SVM:** the original script is not in the repo. None of 96 repo-implied setups reproduced these rows (e04). They were rebuilt following §6.7.1-6.7.2 (e05): English uses word 1-3-gram TF-IDF, Arabic uses word + char TF-IDF, with SMOTE and the same split.
  - Reported is the run whose accuracy is closest to Table 8. Each is within 0.3 points.
  - Macro-F1 is stable across runs within ±1 point.
- **English Decision Tree:** 76.5% could not be reached by any of about 200 setups (max 66.5%, e05b).
  - The authors chose to replace it with the default DT under the paper's setup: 64.77% accuracy / 0.648 macro-F1 (gini, no depth limit).
  - This changes no claim: the order DT < SVM < Stacking still holds in both languages.

## 3. Polarity with macro-F1

**English** (e06; Amazon Review Polarity, fastText English filter → 3,580,535 / 397,826, identical to the paper):

| Model | 3-fold CV acc (paper) | CV acc (rerun) | CV macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.8747 | 0.8742 | 0.8742 |
| LinearSVM | 0.8753 | 0.8753 | 0.8753 |
| Multinomial NB | 0.8205 | 0.8205 | 0.8205 |

Final LinearSVM on the test set: accuracy 0.8767 and macro-F1 0.8767, matching the paper exactly.

**Arabic** (e02; 66,666 balanced):

| Model | 3-fold CV acc (paper) | CV acc (rerun) | CV macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.8679 | 0.8675 | 0.8675 |
| LinearSVM | 0.8614 | 0.8608 | 0.8608 |
| Multinomial NB | 0.8582 | 0.8580 | 0.8580 |

Macro-F1 equals accuracy for polarity because both datasets are almost exactly class-balanced. It is worth saying this in the paper.

## 4. Leakage-safe Arabic polarity (e02)

- **Method:** the 80/20 split is made first. TF-IDF sits inside a Pipeline and is refit in each training fold. Model comparison and GridSearch use the training split only, and the test split is used once.
- **CV (train only):** LogReg 0.8638, LinearSVM 0.8576, MNB 0.8542 (macro-F1 equal).
- **Test** (LogReg, C=2, 53,332 train / 13,334 test): **accuracy 0.8738, macro-F1 0.8738**, against the paper's 0.8742. The leakage had no material effect.

## 5. Smart Router (no rerun, see D1)

- **What the reviewer asked:** clarify whether 88.6% is test accuracy and explain how it differs from 83.92%. This is answered by clarification; nothing needs to be run.
- **The paper's own evidence:**
  - Table 2 labels 88.58% "Training Fit".
  - §6.1 and §6.1.2 say 88.6% was obtained "after retraining on the whole dataset".
  - Fig. 2 covers all 4,720 balanced samples.
  - So 88.6% is the full-data fit, and the held-out estimate is the 83.92% CV accuracy.
- **To confirm with the teammate who ran it:** whether a separate test split existed. Earlier it was stated that 88.6% is a test accuracy.
- **`dataset.csv`:** the router's 5,360 training samples. It is not in the repo (the `.gitignore` excludes `*.csv`). It would be needed only if the reviewer later demands a separate held-out test number.

## 6. English aspect extraction: token-level vs exact-span (r06, r07)

**Data.** SemEval-2014 Task 4 **Laptop + Restaurant** train sets. That is 6,086 sentences, and the official v2 files come from the Hugging Face mirror (`alexcadillon/SemEval2014Task4`).
- Fig. 3 covers 101,005 tokens. Laptop + Restaurant with the repo's regex tokenizer gives 100,861 tokens (0.14% off). Either domain alone gives about half that (r06).

**Model.** The team's tagger: `english/features_en.py` features, `DictVectorizer`, and `LinearSVC` with the same GridSearch and 80/20 token split as `english/training_with_grid_search_en.py`. Aspects of all polarities are tagged (r07).

| | Paper | Rerun, token-level |
|---|---|---|
| B-ASP P / R / F1 | 0.80 / 0.75 / 0.77 | 0.79 / 0.75 / 0.77 |
| I-ASP P / R / F1 | 0.73 / 0.68 / 0.70 | 0.73 / 0.50 / 0.60 |
| O P / R / F1 | 0.99 / 0.99 / 0.99 | 0.98 / 0.99 / 0.98 |
| Overall P / R / F1 | 0.8347 / 0.7803 / 0.8066 | 0.8344 / 0.7466 / 0.7830 (macro over B, I, O) |
| Token accuracy | 97% | 96.1% |

**Conclusion: F1 = 0.8066 is token-level.** Specifically, it is P/R macro-averaged over the three BIO tags (B-ASP, I-ASP, O), with F1 = 2PR/(P+R).
- The B-ASP row and the overall precision are reproduced at token level.
- The same model scored by **exact span** (sentence-level 80/20 split) gets only **P 0.680 / R 0.686 / F1 0.683**.
- I-ASP recall is not reproduced (0.50 vs 0.68): the team's exact preprocessing differs slightly. This does not affect the conclusion.

**Optional for the paper:** report the exact-span F1 of 0.683 alongside it as the stricter SemEval-style measure.

## 7. Latency: what the comment needs (no experiment)

- **Hardware of Table 15:** AMD Ryzen 7 4700U with Radeon Graphics (8 cores / 8 threads, 2.0 GHz base clock), 15.4 GB RAM, Windows 11 Home 10.0.26200. CPU only.
- **LLM calls are not included.** From the code (`extras/benchmark_latency.py`): Table 15 times only the local classical models. The online Qwen dialect→MSA call is excluded.
- **BERT:** the system does not use BERT. §6.9 compares against *published* figures (BERT-base 25-40 ms, DistilBERT 15-25 ms, refs [14], [17]), which were not measured on the same hardware. The reviewer asks to remove this comparison unless it is measured under comparable conditions.
- **The e03 run below is only a sanity check.** It shows the Table 15 magnitudes are reproducible on a laptop CPU. It does not replace the paper's numbers.

### e03 sanity rerun

- **Hardware:** AMD Ryzen 7 7445HS (6 cores / 12 threads), 15.3 GB RAM, Windows 11 Home, Python 3.10.11. Single-threaded, CPU only, no GPU.
- **Protocol:** 105 reviews per pipeline, 5 warm-up runs excluded.
- **The LLM is not included** in any stage or total.

| Stage (ms, mean) | Paper | Rerun |
|---|---|---|
| Arabic: language detection | 0.13 | 0.13 |
| Arabic: dialect routing | 0.79 | 0.82* |
| Arabic: aspect extraction | 0.49 | 0.56 |
| Arabic: aspect classification | 5.02 | 5.87 |
| **Arabic total** | **6.44** | **7.38** |
| English: language detection | 0.15 | 0.14 |
| English: aspect extraction | 0.82 | 0.89 |
| English: aspect classification | 2.51 | 2.21 |
| **English total** | **3.48** | **3.25** |

\*The routing stage is a timing proxy with the same router architecture, because `dataset.csv` is unavailable. This is fine for a sanity check. Nothing further is needed for #7: the hardware is recorded (D6), and timing the LLM separately is not required by the reviewer.

## 8. Arabic datasets: routing vs ABSA, and test leakage (from code)

Each model is trained and tested on its own dataset, with its own split. No model is evaluated on another model's data, so text shared *between* datasets cannot leak. The only question is whether each model's own test split reaches its own training.n

| Model | Arabic data | Own-test leakage (from code) |
|---|---|---|
| Smart Router | `dataset.csv`: LABR + HARD samples labelled MSA/DIALECT by an LLM | TF-IDF is fit on all rows before CV. This affects only the 83.92% CV figure |
| Aspect extraction | SemEval-2016 Arabic hotels (`Arabic_Hotels_TrD_V2.xml`) | ⚠ The 80/20 split is per **token**, so tokens of one sentence fall in both train and test. The `DictVectorizer` is fit on all tokens |
| Aspect sentiment | SemEval-2016 Arabic hotels (`Arabic_Hotels_Cleaned.xml`) | ✅ none: TF-IDF and SMOTE are fit on the training split only |
| General polarity | Arabic 100K Reviews (66,666 after removing Mixed) | The original fit TF-IDF and GridSearch on all rows. **Addressed:** the leakage-safe rerun gives 0.8738 vs the paper's 0.8742 (§4) |

**Aspect extraction: the token-level split leaks in method, but the effect is negligible (e08).** The same models were rerun with a **sentence-level** split, so whole sentences go to train or test and the vectorizer is fit on train only:

| | Token-level split (paper) | Sentence-level split | Change |
|---|---|---|---|
| Arabic macro-F1, 5-fold CV | 0.7104 | 0.7073 | −0.003 |
| Arabic macro-F1, 80/20 hold-out | 0.7165 | 0.7206 | +0.004 |
| English macro-F1, 5-fold CV | 0.7877 | 0.7855 | −0.002 |
| English macro-F1, 80/20 hold-out | 0.7830 | 0.7682 | −0.015 (single-split variance) |

- **Conclusion:** the paper's aspect-extraction numbers are not materially inflated. By 5-fold CV the difference is at most 0.3 points.
- **Option for the paper (authors' decision):** keep the reported numbers and add one sentence reporting the sentence-level check. This lets the paper truthfully say that its results hold under a leakage-free split, as the reviewer asks.

## Reproductions used as the baseline for the above

Everything below matches the paper (differences of ≤0.05 points come from library versions):
- **Arabic aspect sentiment** (r01): 9,826 samples, Table 9 counts identical, accuracy 0.9013 against 0.9018.
- **English aspect sentiment** (r03): Table 10 identical, accuracy 0.8017 against 0.800.
- **Arabic aspect extraction** (r04): token counts are identical (106,876 = 97,235 / 8,139 / 1,502). Held-out macro-F1 0.7165 against 0.7163. The 5-fold folds and the ablation table are identical.
- **Arabic polarity** (r02): 0.8675 / 0.8608 / 0.8580 CV, test 0.8747 against 0.8742.
- **English polarity** (e06): see §3.

## Open items

**Experiments:** none. All 8 reviewer items are resolved.

**Before updating the paper, the authors still need to:**
- **Router (D1):** confirm with the teammate who ran it whether 88.6% is the full-data training fit (as the paper's Table 2 says) or a test accuracy (§5).
- **Aspect extraction leakage:** the sentence-level check is done (§8, e08) and the effect is negligible. Decide whether to add the one-sentence statement to the paper.

## Data and files

| What | Where |
|---|---|
| Experiment scripts | `experiments/reviewer_revision/r0*.py` (reproductions), `e0*.py` (new experiments) |
| Results (JSON) | `experiments/reviewer_revision/results/` (logs and rebuilt intermediate data in `logs/`, `data/`, git-ignored) |
| Amazon Review Polarity (download) | `data/raw/amazon_polarity/` (fast.ai mirror; git-ignored, see `data/README.md`) |
| SemEval-2014 Laptop + Restaurant train v2 | `data/raw/semeval2014_laptops_train_v2.xml`, `data/raw/semeval2014_restaurants_train_v2.xml` (restaurant from Hugging Face `alexcadillon/SemEval2014Task4`) |
| English aspect-sentiment data (was `final_dataset.jsonl`, branch `main` only) | `data/raw/amazon_absa_gpt41_labeled.jsonl` |
| Python environment | `requirements.txt`; the runs here used scikit-learn 1.6.1, imbalanced-learn 0.13.0, fasttext-wheel, spacy 3.8 (spaCy only for r06) |
| **Paper LaTeX source** (where D1–D8 will be applied) | private repo [`ahmedred6/dual-language-absa-paper`](https://github.com/ahmedred6/dual-language-absa-paper), local `paper_latex/`. Build with XeLaTeX; local compiler at `tools/tectonic/tectonic.exe -X compile main.tex` |

Reviewer comments that can be answered from the code alone are documented in `CODE_FINDINGS.md`. Where that file conflicts with the decisions log above, the decisions log takes precedence.
