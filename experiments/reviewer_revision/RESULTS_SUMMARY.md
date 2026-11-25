# Reviewer revision: experiment results

Last updated: 2026-10-25. The paper's facts and numbers stay as they are. This file covers only the reviewer comments that need (re-)experimentation.
Every number here was produced by a script in this folder (`experiments/reviewer_revision/`). The script's output is in `results/<script>.json`. Run the scripts from this folder.
Environment: Python 3.10.11, scikit-learn 1.6.1, imbalanced-learn 0.13.0. Seed 42 throughout.

## Decisions log (apply to the paper later)

| # | Date | Decision | Where in paper |
|---|---|---|---|
| D1 | 2026-10-25 | **Smart Router: no rerun.** Answer the reviewer by clarification. The paper itself describes 88.6% as the fit after retraining on the full dataset: Table 2 "Training Fit 88.58%"; §6.1 and §6.1.2 "retraining on the whole dataset"; Fig. 2 sums to the full 4,720 balanced samples. The held-out estimate is the **83.92%** 5-fold CV accuracy. No number changes. **Tentative (2026-11-14): use the paper's own wording** (training fit; held-out = 83.92% CV). If the teammate confirms a separate test set, swap in one sentence plus the split size | Table 2, §6.1, §6.1.2 |
| D2 | 2026-10-25 | **(Superseded by D22.) Replace the English Decision Tree accuracy in Table 8**: 76.5% → **64.8%**, with macro-F1 **0.648** (e05: default DT under the §6.7.1-6.7.2 setup). No claim changes (DT < SVM < Stacking still holds) | Table 8 |
| D3 | 2026-10-25 | **Add macro-F1 to Table 8** for all rows (values in §2 below) | Table 8 |
| D4 | 2026-10-25 | **Add macro-F1 to the polarity tables** (values in §3); note that macro-F1 equals accuracy because the classes are balanced | Tables 11-14 |
| D5 | 2026-10-25 | **Remove the BERT/DistilBERT latency comparison.** The system does not use BERT, and the figures were not measured on the same hardware. Delete the sentences "For context, published CPU inference benchmarks report approximately 25-40 ms per sentence for BERT-base [14] and 15-25 ms for DistilBERT [17] under comparable single-threaded conditions. Both proposed pipelines thus operate at roughly 4-7 times lower latency than a distilled transformer…". Keep the BERT mentions in Related Work and future work: they are background, not claims of use | §6.9 |
| D6 | 2026-10-25 | **Latency:** state the hardware and that the LLM call is excluded. Table 15 hardware (from the machine that ran it): **AMD Ryzen 7 4700U** (8 cores / 8 threads, 2.0 GHz base), **15.4 GB RAM**, **Windows 11 Home** (10.0.26200), CPU only | §6.9 |
| D7 | 2026-10-25 | **English aspect-extraction F1 0.8066 is token-level** (macro over B-ASP/I-ASP/O). State this; optionally add exact-span F1 0.683 | §6.2 |
| D8 | 2026-10-25 | **SMOTE wording.** Keep all reported (with-SMOTE) numbers. Replace the claim that negative recall would "likely collapse" without SMOTE with the measured comparison. Proposed text: *"To assess the contribution of SMOTE, the Arabic stacking ensemble was trained with and without oversampling on the same split. Overall performance was statistically indistinguishable (accuracy 90.13% vs 90.28%, macro-F1 0.894 vs 0.898; McNemar p = 0.83), because all base learners already apply class-balanced weighting. SMOTE instead shifts the balance of errors between classes (negative recall 86.4% vs 92.8%, positive recall 92.3% vs 88.8%). For the English data, which is naturally balanced, SMOTE had no effect."* (e01, e07) | §6.7.2 (and the general claim in §5.3) |
| D9 | 2026-11-14 | **Aspect-extraction leakage: add one sentence to the Results** reporting the sentence-level check. Keep Tables 3, 5, 6 as they are. Proposed: *"Re-evaluating with a sentence-level split, so that no sentence contributes tokens to both training and test sets, gives a 5-fold macro-F1 of 0.7073 (vs. 0.7104), confirming that the token-level split does not inflate the results."* (e08) | §6.5 / §6.2 |
| D10 | 2026-11-14 | **SOTA comparison table (Discussion):** the user looks up the published numbers and pastes the source text; Claude checks and fills the table. No numbers from memory. | §7 |
| D11 | 2026-11-14 | **Architecture figure replaced** (Architecture comment: "Replace the current figure with a professional diagram ... show dialect normalization explicitly"). New TikZ vector diagram `paper_latex/figures/architecture_diagram.tex` (+ compiled PDF), approved by the user; follows the reviewer's exact flow, Qwen (not Llama 3) for dialect normalization, MSA bypasses the LLM, no-aspect fallback shown. **Applied** to main.tex (Fig. 1). | Fig. 1 (Section 3) |
| D12 | 2026-11-18 | **Remove the CAMeL Tools normalization claims.** The code never uses CAMeL (no import on any branch or in git history). The normalization behind Tables 5 and 7 is the regex `normalize_arabic()`, which §5.1 already describes. Delete or reword the two sentences that credit CAMeL with the normalization: §3 Arabic pipeline (*"Arabic text undergoes preprocessing using CAMeL Tools to standardize …"*) and §6.6 ablation (*"… together with CAMeL-based preprocessing"*). **Keep** the rule-based CAMeL aspect-extractor baseline (Table 4, Listing 1): it is described as an earlier abandoned attempt, which Hamzah confirms. No numbers change. Reason: removes a paper/code contradiction at no cost to results, and answers the Architecture comment on "which CAMeL Tools normalization operations are applied". Not decided: aligning the §5.1 list with the code (the code also maps ى → ي and does not convert Arabic-Indic digits) | §3, §6.6 |
| D13 | 2026-11-18 | **Remove the spaCy cleaning claim.** The English pipeline does not use spaCy. Cleaning is the regex `normalize_english()` (curly quotes → ASCII, non-ASCII symbols → space, whitespace collapsed), tokenization is the regex `\w+\|[^\w\s]`, and there is no lowercasing at the extraction step (lowercasing is done only inside the token features and for the sentiment stage). spaCy appears only in the abandoned CRF experiment (`archive/english_crf_tagger/`) and `archive/misc/training_snippet.py`, neither of which produced a paper number. Replace the §3 English-pipeline sentence *"The text is first cleaned and lowercased using the spaCy library …"* with the actual cleaning steps. **Keep** the spaCy noun-chunk baseline (§6.2.1, F1 0.259): it is described as an earlier abandoned attempt, as with the CAMeL baseline (D12). No numbers change. Reason: same as D12. Removes a paper/code contradiction at no cost to results, and answers the Architecture comment "state the actual ... text-cleaning steps" | §3 |
| D14 | 2026-11-18 | **Correct the aspect-extraction feature description.** The submitted PDF (pp. 9-10, §5.2) says each token uses word TF-IDF n-grams (plus char TF-IDF for Arabic), POS distributions, punctuation/capitalization indicators and a ±2-token window. The code behind Tables 3, 5, 6, 7 (`token_features()` in `src/3_aspect_extraction/arabic/features.py` and `english/features_en.py`, unchanged since 2026-03-10 and reproduced to 4 decimals in r04/r07) uses `DictVectorizer` + `LinearSVC` over: token, lowercased token, isdigit, prefix-1/2, suffix-1/2, previous and next token (**±1 window**). No TF-IDF, POS, capitalization or punctuation features (POS and ±2 exist only in the abandoned CRF code, `archive/english_crf_tagger/`). The TF-IDF described in §5.2 is the **aspect-sentiment** representation: word TF-IDF 1-3-grams (20k, sublinear) + char_wb TF-IDF 3-5-grams (30k), for **both** languages (not Arabic only). Rewrite §5.2 as (a) extraction features and (b) sentiment features. In §3, replace "English TF-IDF vectorizer and an SVM" / "Arabic-specific TF-IDF vectorizer trained on unigrams and bigrams, together with an SVM-based BIO tagger" with token-level features and an SVM BIO tagger. No numbers or claims change (the contribution "TF-IDF + stacking" refers to sentiment and stays true). Answers the Methodology "Feature Representation" comment and the Architecture "BIO tagger's features" / "actual TF-IDF settings" comments. Note for Table 8: the e05 English DT/SVM baselines used word-only TF-IDF, while the English stacking model uses word + char | §5.2, §3 |
| D15 | 2026-11-18 | **Tighten the Smart Router description (wording only).** The submitted PDF (p. 6, §3.2) says the router "utilizes a classical machine learning model to separate English reviews from Arabic ones and further distinguishes between MSA and Dialectal Arabic". The code (`src/1_router/train_router.py`, originally `FINALWINNERYAY.py`) trains a **2-class** model only (`df['label'].isin(['MSA', 'DIALECT'])`, LABR + HARD, no English data). English vs Arabic is a separate, non-ML step: Arabic-script regex in `whole_pipeline.py`, `lingua` in the dashboard, fastText `lid.176` in the latency benchmark (Table 15). Table 2, Fig. 2 and §4.2 are already 2-class, so only §3.2 changes. Rewrite §3.2 as two stages: (1) a lightweight language check sends Arabic-script reviews to the Arabic branch and the rest to the English branch; (2) the Logistic Regression router (char_wb TF-IDF 3-5-grams, 10k features) separates MSA from dialect. Answers the Architecture comment on the router's "three target categories" (there are two). No numbers change. **Not decided:** which detector to name (recommended: generic "script-based language check", or fastText since Table 15 timed it). **Open, ask Maya:** §6.1 says C / max_depth were tuned by grid search, but the code uses fixed values (LR default C, DT max_depth=20) with plain 5-fold CV | §3.2 |
| D16 | 2026-11-18 | **Describe general polarity as a parallel review-level output, not a conditional fallback (wording only).** The submitted PDF (p. 7, §3.3 and §3.4) says reviews "that yield no extracted aspects" are handled by "a linear SVM General Polarity classifier". The code (`whole_pipeline/whole_pipeline.py: analyze_sentence_global`, same in `Dashboard/pipeline.py`) runs the polarity model on **every** review with no condition and returns it as a separate field `overall_polarity` beside the aspect results (`absa`); with no aspects the extractor returns `"aspects": []`. Also, the Arabic polarity model is **Logistic Regression** (§6.8 and the e02 rerun), not a linear SVM. Proposed text: *"A general-polarity classifier (linear SVM for English, Logistic Regression for Arabic) labels every review as a whole. Its output is reported as a separate review-level field next to the aspect-level results, and it is the only sentiment signal for reviews with no extracted aspects."* No numbers change. Answers the Architecture comment on the fallback's invoking condition and how its predictions are distinguished from aspect-level results. Approved on the condition that the impact is low (confirmed) | §3.3, §3.4 |
| D17 | 2026-11-18 | **Qwen guided summarization (superseded by D17b: no longer waiting on the teammate).** The component exists (authors confirm) but its code was never committed: no summarization code, prompt or output on any branch or in git history; only the dashboard's descriptive text mentions it. The paper describes it on p. 2 (contributions), §3.5, p. 8 (dashboard), p. 21 (Discussion) and p. 22 (Conclusion), and claims it "mitigates"/"prevents" hallucination. **Waiting on the teammate who built it for:** (1) the code or prompt; (2) exact model and temperature; (3) the fields passed to Qwen; (4) whether summaries were ever checked against the statistics. If not evaluated: keep the component, soften "mitigates/prevents hallucination" to "grounded in the aggregated statistics", or run a small consistency check once the code is available (professor: "Do not claim hallucination reduction without evaluation"). Related finding: the submitted reference list has 5 entries printed as "Placeholder entry: verify ..." (Darwish 2021, Zalmout 2019, Abdul-Mageed 2020, Wang 2021, Tang 2024), including [31] Tang 2024, which backs the hallucination claim. For the Literature owner | §3.5, p. 2, §7, Conclusion, references |
| D17a | 2026-11-18 | **Backup implementation of the guided summarizer, built to the paper's §3.5 description** (the teammate's original stays preferred if recovered). `src/6_pipeline/summarizer.py`: (1) aggregate aspect results from both pipelines into one table (language, aspect, positive, negative), grouping identical terms across reviews; English and Arabic terms kept as separate rows (the paper does not claim translation); (2) send only that table to Qwen with the repo's existing settings (`qwen-plus-2025-04-28`, DashScope endpoint, temperature 0, max_tokens 3000); (3) automatic consistency check: the model also returns its claims as JSON and every aspect, count and sentiment is verified against the table. Shown on a new dashboard page. The paper text is not changed; wording about the check and any hallucination claim is decided separately once it has been run | §3.5 |
| D17b | 2026-11-18 | **The summarizer in `src/6_pipeline/summarizer.py` is the paper's summarization component**; the teammate's original is no longer needed. It matches every statement about summarization in the paper (p. 2, §3.1, §3.5, §7, Conclusion, Fig. 1; checked 2026-11-18). The paper reports no summarization outputs or numbers, so nothing depends on the original. Any example summary shown in the paper must be generated with this implementation. **Still open:** (a) whether to run it (one demo run and/or a small evaluation of the consistency check; needs `QWEN_API_KEY`); (b) the hallucination wording ("mitigates"/"prevents" in §3.5 and §7): keep only if the evaluation supports it, otherwise soften to "grounded in the aggregated statistics" | §3.5, §7 |
| D18 | 2026-11-18 | **Dialect-normalization LLM: Qwen, as the submitted paper states (p. 7, §3.4: "Qwen, a large language model, is employed as a semantic normalizer"). No change to the paper.** The new Fig. 1 (D11) already says Qwen, so the old figure's "Llama 3" label is gone. For the record only: in the repo, `dialect_to_msa.py` and `whole_pipeline.py` have GPT-4.1 active (the Qwen client in `dialect_to_msa.py` is commented out), and the dashboard uses `qwen-plus-2025-04-28`. The authors' decision is to follow the paper | §3.4, Fig. 1 |
| D19 | 2026-11-18 | **Table 8 baselines: protocol fixed BEFORE running (Hamzah's review point #4), script `e10_table8_fair_baselines.py`.** Replaces the "closest to the paper's number" selection of e05. For each language: (1) same data and 80/20 stratified split (seed 42) as the Stacking row; (2) the **same features as Stacking**: `get_advanced_vectorizer()` (word TF-IDF 1-3 + char_wb 3-5), fit on training data only, for **both** languages; (3) SMOTE on the training data for all three models, as for Stacking; (4) tuning: Decision Tree and Linear SVM get a small grid chosen by 5-fold CV **on the training split only** (DT: max_depth {None, 20, 50} × min_samples_leaf {1, 5}; SVM: C {0.1, 0.5, 1, 2}); Stacking keeps the paper's fixed configuration (untuned, so the comparison is conservative); (5) test set scored once; report accuracy + macro-F1, 5-fold stratified CV mean ± std (whole pipeline refit per fold), and McNemar exact tests DT vs Stacking and SVM vs Stacking on the test set. Whatever comes out is reported; no rerun to approach the paper's numbers. **Whether these replace the Table 8 DT/SVM rows is decided after the run** | Table 8 |
| D20 | 2026-11-18 | **Leave the English aspect-extraction confusion matrix (submitted Fig. 3) out of the paper.** It was already removed in the 2026-11-14 Results rewrite (shortening; professor: keep only the baseline-vs-SVM comparison). It was also inconsistent: it shows aspect recall 0.338 over all 101,005 tokens (the whole dataset), while Table 3 gives B-ASP recall 0.75; e09 confirms ≈0.74 on a proper sentence-level test split, so Table 3 is correct and the old figure came from a different run. Table 3 is unchanged; no replacement figure (preview `results/figures/e09_fig3_preview.png` kept for reference only) | §6.3 (Fig. 3 removed) |
| D21 | 2026-11-20 | **Report exact-span F1 for both languages beside the existing token-level scores (Hamzah's point #2, option a).** The token-level numbers (English 0.8066, Arabic 0.7163 / 0.7104) and Tables 3, 5, 6 stay unchanged. Source: e09, sentence-level 5-fold CV, same models. (1) §6.3 English: replace *"a re-evaluation of the same tagger on a sentence-level 80/20 split gives F1 = 0.683"* with *"a re-evaluation of the same tagger with sentence-level 5-fold cross-validation gives F1 = 0.693 ± 0.011"* (0.683 was the first, single-split check in D7). (2) §6.4 Arabic, add after the sentence-level sentence: *"Under the stricter exact-span criterion, the same tagger reaches F1 = 0.622 ± 0.013 (sentence-level 5-fold cross-validation)."* B/I-only macro-F1 is not reported (recorded in e09 only) | §6.3, §6.4 |
| D22 | 2026-11-20 | **Table 8: report the fair baselines (e10, protocol D19) and soften the stacking claim (Hamzah's point #4, option a). Supersedes D2.** (1) Table 8 DT and SVM rows become: English DT **63.3% / 0.633**, SVM **79.9% / 0.799**; Arabic DT **83.1% / 0.820**, SVM **90.1% / 0.896**. Stacking rows unchanged (80.0% / 0.802; 90.2% / 0.894; e10 reproduces them at 80.2% and 90.1%). Bold the best value per column (Arabic macro-F1: SVM 0.896 vs Stacking 0.894, a tie). (2) §6.5: replace *"English uses word-level TF-IDF with 1--3 n-grams; Arabic adds character-level TF-IDF (3--5 n-grams), which improved accuracy by approximately 4 percentage points in preliminary experiments"* with the actual features (both languages: word TF-IDF 1-3-grams + character TF-IDF 3-5-grams, as in D14), and replace *"It achieves the best accuracy and macro F1 in both languages; in a five-fold stratified comparison on English it also had the highest mean accuracy (0.7922)"* with: *"All three models use the same features, split and SMOTE; the Decision Tree and SVM were tuned by cross-validation on the training data only. Stacking clearly outperforms the Decision Tree (McNemar p < 10⁻¹⁴ in both languages), but a tuned linear SVM performs on par with it (p = 0.42 English, p = 1.0 Arabic) at a fraction of the training cost."* Add one sentence that the baselines were retrained for this revision. (3) §3.3/§3.4 stacking description: drop the unsupported *"mitigating individual estimator biases and producing more stable aspect-sentiment classifications"* (also flagged by the professor's "avoid unsupported claims such as 'more stable and accurate'"). Contribution (p. 2), SOTA table rows and Conclusion stay: they claim competitive accuracy at low cost, which still holds | Table 8, §6.5, §3.3, §3.4 |
| D23 | 2026-11-20 | **Evaluate the LLM dialect normalization (Hamzah's point #3): protocol fixed before running, script `e11_llm_dialect_normalization.py`.** (1) **Polarity model:** the leakage-safe Arabic model from e02 (word TF-IDF 1-2-grams, 50k, + Logistic Regression C=2, `normalize_arabic` preprocessing), trained on the 80/20 training split (seed 42) only. (2) **Sample:** 2,000 reviews drawn at random, stratified by label (seed 42), from the **test split only**. (3) **Dialect identification:** zero-shot LLM annotation (the strategy of §4.2), Qwen `qwen-plus`, temperature 0, 20 reviews per call, labels MSA / DIALECT. No router is used (the router data is not available). (4) **Normalization:** each DIALECT review is rewritten once by Qwen `qwen-plus`, temperature 0, with the pipeline's own prompt (`MSA_PROMPT` in `src/6_pipeline/whole_pipeline.py`, verbatim). MSA reviews are not rewritten (they bypass the LLM, as in the pipeline). (5) **Comparison** on the DIALECT subset, same reviews, same model: accuracy and macro-F1 on raw vs normalized text, McNemar exact test. Context: accuracy on the MSA subset. (6) **Human check:** 150 random (seed 42) original/normalized pairs exported to CSV for Maya and Hamzah to judge meaning preservation. (7) All LLM replies are cached, so an interrupted run resumes without new calls. Model: `qwen-plus` because the free quota of `qwen-plus-2025-04-28` is exhausted (both are Qwen, consistent with D18); the exact model string returned by the API and the run date are recorded. Whatever comes out is reported. *Amendment (2026-11-20, after a transient HTTP 400 stopped the first run at 620/2,000 labels and before any results were computed):* requests rejected by the API are retried; a rejected batch is re-sent one review at a time; reviews still refused (e.g. by the API content filter) are excluded and their number is reported | §6 (new), §7.4 |
| D24 | 2026-11-20 | **Add the LLM-normalization check (e11) to the paper as evidence that normalization preserves sentiment.** New short Results paragraph (e.g. at the end of §6.6 Polarity, or its own subsection), proposed: *"To check that LLM normalization preserves sentiment, 2,000 reviews from the Arabic polarity test split were labelled as MSA or dialect by zero-shot annotation with Qwen (qwen-plus), and the 1,078 dialectal reviews were classified by the polarity model before and after Qwen normalization. Accuracy did not change significantly (90.0% raw vs. 88.4% normalized; McNemar p = 0.07), and 92.7% of predictions were unchanged. Normalization therefore preserves review-level sentiment; it did not improve accuracy for this classifier, which was trained on dialect-rich data."* Optionally one clause in §7.4 Limitations: the check used the review-level polarity classifier, since no labelled dialectal ABSA data exist. Claims kept as they are: "preserving sentiment structure" (now supported) and the router asymmetry observation (professor asked to keep it). The paper must not claim that normalization improves accuracy | §6.6, §7.4 |
| D25 | 2026-11-20 | **Transformer latency comparison on the same CPU (Hamzah's point #5, option b): protocol fixed before running, script `e12_transformer_latency.py`.** Machine: this laptop (AMD Ryzen 7 7445HS), reported as such; **Table 15 (Ryzen 7 4700U) is not changed**; the result is a separate relative comparison. (1) **Same reviews and protocol as e03:** same 105 sampled reviews per language (seed 42), first 5 warm-up excluded, batch size 1, CPU only. (2) **Ours:** the e03 classical pipeline, re-timed in the same session. (3) **Transformer-equivalent ABSA pipeline** doing the same work per review: one token-classification forward pass over the review (aspect extraction) + one sequence-classification forward pass per aspect (aspect window), with the number of aspects equal to what our extractor found on that review, so both pipelines classify the same aspects. Tokenization is included in the timing. Models: English `bert-base-uncased` and `distilbert-base-uncased`; Arabic `aubmindlab/bert-base-arabertv02` (BERT-base size). Classification heads are randomly initialized (weights do not affect inference cost); no fine-tuning, so only latency is reported, not accuracy. (4) **Threads:** primary comparison single-threaded for both (as e03); secondary: transformers with all CPU threads (their best case). (5) Report mean / std / p95 ms per review for the ABSA stages (extraction + classification) and the slowdown ratio transformer / ours. Language detection, routing and the LLM call are excluded from both sides. Whatever comes out is reported | §6.7, contribution (p. 2) |
| D26 | 2026-11-20 | **Add the same-CPU transformer comparison (e12) to the latency section.** Table 15 unchanged. Add a short paragraph and table in §6.7: on one machine (Ryzen 7 7445HS), same reviews and same aspects, ABSA stages only: ours 5.6 / 5.5 ms per review (EN / AR) vs DistilBERT 114 ms (1 thread) / 63 ms (12 threads), BERT-base 231 / 120 ms, AraBERT 310 / 212 ms, i.e. 11-56x slower; latency only (transformer heads not fine-tuned). Replaces the removed published-figure comparison (D5) with one measured under comparable conditions. Update the SOTA table's English span F1 to 69.3 and add the Arabic span F1 62.2 (D21) | §6.7, Table 10 |
| D27 | 2026-11-20 | **Evaluate the guided summarizer's grounding: protocol fixed before running, script `e13_summarizer_grounding.py`.** Input: real reviews with their **gold** aspect labels (so extraction errors do not mix in): English from `amazon_absa_gpt41_labeled.jsonl`, Arabic from SemEval-2016 hotels (`NULL` aspects dropped; positive / negative only, as in the binary pipeline). 25 batches per language, each of 30 random reviews with at least one aspect (seed 42). Each batch goes through `summarizer.py` unchanged: aggregation, Qwen `qwen-plus` at temperature 0, consistency check. Report: share of summaries passing the check; count and type of errors (invented aspect, wrong count, flipped sentiment, unsupported number, mentioned-but-unclaimed aspect, invalid JSON). Whatever comes out is reported; it decides how the paper words the hallucination claims (D17b). *Amendment (2026-11-20, after run 1, approved by the author):* run 1 is kept and reported; run 2 uses a corrected checker (digits inside aspect names, e.g. "MP3 player", are not read as numbers; Arabic aspect names are matched after orthographic normalization) and a tightened prompt (at most 8 aspects; every aspect named must be listed in the claims), with full summaries saved. Same batches, same model and temperature. Both runs are reported | §3.5, §7 |
| D28 | 2026-11-20 | **Remove the hallucination claims about our summarizer (professor: "Do not claim hallucination reduction without evaluation"); e13 stays in the records only.** Applied to `main.tex`: §3.5 *"By grounding the generative output in structured statistical data, this approach mitigates the hallucination and factual drift commonly observed in unguided abstractive summarization"* → *"The summary is generated from these aggregated statistics only, not from the raw review text"*; Conclusion *"while preserving factual grounding in the extracted aspect-sentiment distributions"* → *"based solely on the extracted aspect-sentiment distributions"*. The literature sentence in §2 (other work, cited) is unchanged; its citation [Tang 2024] is a placeholder for the Literature owner. Supersedes the open wording question in D17b | §3.5, Conclusion |
| D29 | 2026-11-23 | **Smart Router held-out test and topic check (Hamzah's point #1): protocol fixed before running, script `e14_router_heldout.py`.** Data: Maya's `router_dataset.csv` (verified by r08). (1) **Held-out test:** same label filtering and class balancing as `train_router.py` (downsample DIALECT to 2,360, seed 42); stratified 80/20 split (seed 42) **before** any fitting; Pipeline(char_wb TF-IDF 3-5-grams, 10k features + LogisticRegression(class_weight="balanced", max_iter=1000)), the paper's winning configuration, untuned; 5-fold stratified CV on the training split only; fit on the training split, test set scored once: accuracy, macro-F1, per-class precision / recall, confusion matrix. (2) **Topic check:** each review's domain (BOOK = LABR, HOTEL = HARD, OTHER) labelled zero-shot by Qwen `qwen-plus` (temperature 0, 20 per call, cached). Report: label distribution per domain; accuracy of a domain-only baseline (predict the majority label of each domain, fit on train, scored on test); router test accuracy **within** each domain. If the router is clearly above the domain-only baseline and accurate within each domain, it is learning dialect, not topic. Whatever comes out is reported | §6.2, §4.2 |
| D30 | 2026-11-24 | **Report the router held-out test and the topic confound (e14, option a). Applied to `main.tex`.** §6 router paragraph: added *"To obtain a true test estimate, the same configuration was retrained on a stratified 80% split and evaluated once on the remaining 944 reviews, reaching 83.3% accuracy (macro F1 0.833), consistent with the cross-validation estimate."* Table 2 and the 88.6% / 83.92% figures unchanged. §7.4 Limitations, first limitation extended (still four limitations, as the professor asked): LLM labels **and** dialect confounded with topic (hotels mostly dialect, books mostly MSA), so the router partly relies on topic (held-out: 11% of MSA hotel reviews and 60% of dialectal book reviews recognized). Future work: human-validated, domain-balanced router data. Open for Maya: §6 says the router models were compared "using Grid Search", but `train_router.py` uses fixed settings with plain 5-fold CV | §6.2, §7.4, §8 |
**Applied to the paper (2026-11-20):** D21, D22, D24 and D26 were applied to `paper_latex/main.tex` (§3.3, §3.4 stacking wording; §6.3, §6.4 exact-span; §6.5 text and Table 6; §6.6 normalization paragraph; §6.7 new Table 10; Table 11 SOTA rows; §7.1 and §7.3 discussion). Consequence of D21 in §7.1: the Arabic comparison now reads that the Bi-LSTM-CRF (69.98, most likely exact-span) is ahead of our exact-span F1 (62.2), replacing "of a similar order to our token-level scores". Paper now 20 pages. **D12-D16 applied 2026-11-22** (§3.2, §3.3, §3.4, §5.2; D12 §5.1 list corrected 2026-11-22: Alef Maqsura added, digit conversion removed, stopwords only for English general polarity, TF-IDF vocabularies capped; D15 detector left generic: "script-based language check").

**Applied to the paper (2026-11-14):** §6 Results and §7 Discussion of `paper_latex/main.tex` were rewritten applying D1–D10 (Results 440 → 252 lines incl. Discussion, ~43% shorter; paper 24 → 19 pages). Highlighted diff: `paper_latex/main_diff.pdf`.

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

## e10. Table 8 baselines under a pre-registered protocol (D19, run 2026-11-18)

Protocol fixed before running (D19): same split, the same word + char TF-IDF features and SMOTE as Stacking for both languages; DT and SVM tuned by 5-fold CV on the training split only; Stacking untuned (paper configuration); test set scored once. Results in `results/e10_table8_fair_baselines.json`.

| | Paper acc | Test acc / macro-F1 | 5-fold CV acc (mean ± std) | Chosen params |
|---|---|---|---|---|
| English DT | 76.5% | 63.3% / 0.633 | 65.1% ± 0.5 | max_depth 50, min_samples_leaf 1 |
| English SVM | 78.2% | 79.9% / 0.799 | 79.4% ± 0.5 | C = 0.5 |
| English Stacking | 80.0% | 80.2% / 0.802 | 79.9% ± 0.4 | paper configuration |
| Arabic DT | 71.2% | 83.1% / 0.820 | 82.3% ± 0.8 | max_depth None, min_samples_leaf 1 |
| Arabic SVM | 85.5% | 90.1% / 0.896 | 90.4% ± 0.7 | C = 0.1 |
| Arabic Stacking | 90.2% | 90.1% / 0.894 | 90.4% ± 0.7 | paper configuration |

McNemar exact tests on the test set: SVM vs Stacking English 57 vs 67 discordant, **p = 0.42**; Arabic 39 vs 39, **p = 1.0**. DT vs Stacking: p < 1e-14 in both languages.

**Conclusion (finding, not a decision):** Stacking reproduces the paper. With the same features and equal tuning, a single Linear SVM is statistically indistinguishable from Stacking in both languages (gain 0.3 and 0.0 points vs the paper's 1.8 and 4.7). Stacking still clearly beats the Decision Tree. The paper's DT/SVM rows used weaker features or settings than Stacking. **How to handle this in the paper is pending the authors' decision.**

## e09. Aspect extraction: exact-span and B/I-only metrics (Hamzah's point #2, run 2026-11-18)

Same models and features as the paper. Sentence-level split = whole sentences in train or test, vectorizer fit on train only, identical sentence texts kept together. Results in `results/e09_ate_span_metrics.json`.

| | Arabic | English |
|---|---|---|
| Paper: token-level macro-F1 over B/I/O (token split) | 0.7163 | 0.8066 |
| Rerun, token split: macro-F1 B/I/O | 0.7165 | 0.7830 |
| Rerun, token split: macro-F1 **B/I only** | 0.593 | 0.684 |
| Sentence split hold-out: **exact-span P / R / F1** | 0.565 / 0.727 / **0.636** | 0.710 / 0.702 / **0.706** |
| Sentence split hold-out: macro-F1 B/I only | 0.599 | 0.695 |
| Sentence split hold-out: per-tag F1 B / I / O | 0.686 / 0.512 / 0.964 | 0.771 / 0.619 / 0.983 |
| Sentence split 5-fold CV: exact-span F1 (mean ± std) | 0.622 ± 0.013 | 0.693 ± 0.011 |
| Sentence split 5-fold CV: macro-F1 B/I only | 0.577 ± 0.013 | 0.687 ± 0.009 |
| Grouped-tuned (grid on training sentences only): exact-span F1 | 0.649 | 0.710 |

- **O inflates the paper's metric by about 0.10-0.12** (B/I/O vs B/I-only macro-F1 on the same predictions).
- **Exact-span F1 is about 0.62-0.65 (Arabic) and 0.69-0.71 (English).** The earlier English figure of 0.683 (r07, D7) used a plain sentence split without duplicate grouping; it lies within this CV range.
- Tuning on training sentences only changes exact-span F1 by ≤ 0.013, so the paper's settings are not over-tuned.
- **Fig. 3:** a confusion matrix of the same model on the sentence-level test split gives aspect recall **0.743** / precision 0.865 (19,740 test tokens), consistent with Table 3 (B-ASP recall 0.75). The submitted Fig. 3 (recall 0.338, 101,005 tokens = the whole dataset) does not come from the Table 3 model or test split. Preview only, not inserted: `results/figures/e09_fig3_preview.png`.
- **How to report these in the paper is pending the authors' decision** (no paper numbers changed).

## e11. Does the LLM dialect normalization help? (Hamzah's point #3, protocol D23, run 2026-11-20)

Qwen `qwen-plus` (temperature 0). Polarity model: e02 leakage-safe Logistic Regression trained on the training split (full test accuracy 0.8738, = e02). Sample: 2,000 test-split reviews, stratified. Qwen labelled 1,084 DIALECT (54.2%) and 916 MSA; 0 labelling requests and 6 rewrites were refused by the API (excluded). Results in `results/e11_llm_dialect_normalization.json`; cached LLM replies in `results/e11_cache/`.

| Dialect reviews (n = 1,078) | Accuracy | Macro-F1 |
|---|---|---|
| Raw dialect text | **90.0%** | 0.899 |
| After Qwen normalization to MSA | 88.4% | 0.883 |
| *Context: MSA reviews, raw (n = 916)* | *86.8%* | *0.866* |

McNemar exact: 48 reviews correct only on raw, 31 only on normalized, **p = 0.07** (not significant). The predicted sentiment is unchanged for 999 / 1,078 reviews (92.7%).

**Conclusion (finding, not a decision):** no evidence that LLM normalization improves sentiment accuracy on dialect reviews; the small drop (-1.6 points) is not significant. The classifier is not weaker on dialect (90.0%) than on MSA (86.8%), because Arabic 100K already contains much dialect. Caveat: the classifier was trained on raw text, so normalized text is slightly out of distribution; a fully fair test would also normalize the ~53k training reviews (beyond the free quota). Supports the paper's "preserving the sentiment structure" (no significant change, 92.7% of predictions unchanged), not a claim of improved accuracy. Human meaning check: `results/e11_human_check.csv` (150 pairs, for Maya and Hamzah). **How to report this is pending the authors' decision.**

## e12. Same-CPU latency: our pipeline vs transformers (Hamzah's point #5, protocol D25, run 2026-11-20)

Machine: AMD Ryzen 7 7445HS (6 cores / 12 threads), 15.3 GB, Windows 11, CPU only, torch 2.14.1+cpu, transformers 4.57.6. **Not the Table 15 machine; Table 15 is unchanged.** Same 105 reviews per language as e03 (5 warm-up excluded), batch size 1. Both pipelines classify the same aspects (English 0.83, Arabic 2.25 per review). ABSA stages only (extraction + aspect classification); language detection, routing and the LLM excluded. Transformer heads untrained: latency only, no accuracy. Results in `results/e12_transformer_latency.json`.

| Per review (mean; p95) | English | Arabic |
|---|---|---|
| **Ours** (SVM BIO tagger + stacking), 1 thread | **5.6 ms** | **5.5 ms** |
| DistilBERT-base-uncased (67M), 1 thread | 114 ms; p95 260 (20x) | |
| DistilBERT-base-uncased, 12 threads | 63 ms; p95 161 (11x) | |
| BERT-base-uncased (109M), 1 thread | 231 ms; p95 521 (41x) | |
| BERT-base-uncased, 12 threads | 120 ms; p95 287 (21x) | |
| AraBERT v02 (135M), 1 thread | | 310 ms; p95 634 (56x) |
| AraBERT v02, 12 threads | | 212 ms; p95 465 (39x) |

**Conclusion (finding):** on the same CPU, the classical pipeline is 11-56x faster than transformer equivalents doing the same work; even in the transformers' best case (all 12 threads vs our 1, smallest model) the gap is 11x (English) and 39x (Arabic). Our English time here (5.6 ms) is higher than in e03 on the same laptop (3.2 ms), likely because torch was loaded in the same process; this makes the ratios conservative. Supports the contribution "substantially lower computational cost than transformer-based alternatives" with a same-machine measurement. **How to report it is pending the authors' decision.**

## e13. Summarizer grounding (protocol D27, run 2026-11-20)

50 batches of 30 real reviews with gold aspect labels (25 English, 25 Arabic), summarized by Qwen `qwen-plus` (temperature 0) and checked by `summarizer.check_summary`. Results in `results/e13_summarizer_grounding.json`.

| | English | Arabic |
|---|---|---|
| Summaries passing the strict check | 13 / 25 | 12 / 25 |
| Wrong count | 0 | 0 |
| Flipped sentiment | 0 | 0 |
| Invented aspect | 0 | 1 (الإنترنت; the table has the spelling variant الانترنت) |
| "Unsupported number" | 10 (in 7 batches) | 0 |
| Aspect mentioned in the text but missing from the claims list | 59 | 52 |

**Reading (finding, not a decision):** no summary misstated a count or flipped a sentiment in any of the 50 batches. The failures are of two kinds. (1) **Checker false positives:** digits inside aspect names are read as numbers (verified: English batch 1 failed only on the "3" of the aspect "MP3 player"; the other flagged numbers, e.g. 1394, 1000, 256, look like product names such as "IEEE 1394" or "P1000"), and Arabic matching does not normalize spelling (الإنترنت vs الانترنت). (2) **Incomplete claims lists:** Qwen names more aspects in the text than it lists in its JSON claims, so those mentions are real aspects from the table but their stated sentiment is not verified. The strict pass rate (50%) therefore understates grounding; a corrected checker is needed before a pass rate is reported. Summaries were not saved in full in this run (only 2 examples per language).

### e13 run 2 (D27 amendment: corrected checker, tightened prompt; same batches)

Results in `results/e13_summarizer_grounding_run2.json` (full summaries and claims saved).

| | English | Arabic |
|---|---|---|
| Summaries passing the check | 13 / 25 | 11 / 25 |
| Claims verified against the table | 285 | 259 |
| Wrong count / flipped sentiment / invented aspect / unsupported number | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Aspect mentioned in the text but missing from the claims | 82 | 77 |

**Reading (finding, not a decision):** all 544 verified claims match the statistics exactly, and the corrected checker finds no unsupported number and no invented aspect in 50 summaries. But Qwen does not follow the "at most 8 aspects" rule: it names about 10-15 aspects and lists only part of them as claims, so about half of the summaries contain mentions the check cannot verify. Manual reading of one Arabic summary found two problems outside the check's scope: an incorrect English gloss (فندق rendered as "hotel management") and one aspect placed in two sentiment groups. Pass rate 48% under the strict rule. The paper cannot claim that summaries are free of hallucination; it can claim that every count and sentiment the check could verify (544 claims) was correct.

## r08. Smart Router reproduced with Maya's dataset (2026-11-23)

Maya supplied `dataset.csv` (5,360 rows: 3,000 DIALECT, 2,360 MSA), saved as `data/raw/router_dataset.csv` (git-ignored, like the other data). `src/1_router/train_router.py` run **unchanged** (harness `r08_router.py` only redirects its output files):

| | Paper | Rerun |
|---|---|---|
| Balanced samples | 4,720 | 4,720 |
| LR 5-fold CV accuracy | 83.92% | **83.92%** |
| LR training fit | 88.58% | **88.58%** |
| Fig. 2 (dialect correct / dialect→MSA / MSA→dialect / MSA correct) | 2,068 / 292 / 247 / 2,113 | **identical** |
| DT CV / Ensemble CV | 72.71% / 73.28% | 72.69% / 73.96% (DT has no fixed seed) |

**Conclusion:** this is the dataset behind the paper. D1 is confirmed: 88.58% is the training fit, 83.92% the cross-validation estimate. The held-out test experiment (Hamzah's point #1) is now possible.

## e14. Smart Router: held-out test and topic check (Hamzah's point #1, protocol D29, run 2026-11-23/24)

Results in `results/e14_router_heldout.json`; Qwen domain labels cached in `results/e14_cache/`. (Run 1 stopped at the first Qwen call on a dropped connection; the client now retries on that error. Part 1 is deterministic and identical in both runs.)

**(1) Held-out test** (balanced 4,720; split 80/20 before fitting; paper's configuration, untuned):

| | Accuracy |
|---|---|
| Paper: training fit | 88.58% |
| Paper: 5-fold CV on all data (TF-IDF fit before CV) | 83.92% |
| 5-fold CV on the training split only | 84.08% ± 0.89 |
| **Held-out test (n = 944)** | **83.26%** (macro-F1 0.833; 79 / 79 errors each way) |

The TF-IDF leak did not matter (0.7 points). The router's held-out accuracy is about 83%.

**(2) Topic check.** Qwen domain labels: 2,840 BOOK (LABR), 1,587 HOTEL (HARD), 293 OTHER. Labels are strongly confounded with domain: hotels 86% DIALECT, books 71% MSA. A domain-only baseline (majority label per domain) reaches **74.5%** test accuracy.

| Test set, per domain | n | DIALECT recall | MSA recall | Router accuracy | Majority-class accuracy |
|---|---|---|---|---|---|
| Books | 553 | 0.596 | 0.936 | 0.837 | 0.709 |
| Hotels | 329 | 0.982 | **0.109** | 0.836 | 0.833 |
| Other | 62 | 0.757 | 0.800 | 0.774 | 0.597 |

**Conclusion (finding, not a decision):** the router relies largely on topic: it calls almost every hotel review DIALECT (MSA recall 0.11) and misses 40% of dialectal book reviews. It still carries real dialect signal (books 83.7% vs 70.9% majority; other 77.4% vs 59.7%), but only about 9 points above the topic-only baseline overall. Consequence for the pipeline: dialectal book reviews often skip normalization (the error type the paper calls more damaging). **How to report this is pending the authors' decision.**
