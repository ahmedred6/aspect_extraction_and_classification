# Reviewer comments answered by reading the code

> **Note (2026-10-25):** this file was written before the authors' decisions. Where it conflicts with the
> **decisions log in `RESULTS_SUMMARY.md`**, the decisions log takes precedence. In particular:
> - **D1:** the router needs no rerun; it is answered by clarification (teammate to confirm whether 88.6% is a training fit or a test accuracy).
> - **D5/D6:** latency needs no rerun. The hardware is recorded, and the BERT comparison is removed.
> - **#8:** this is resolved from the code.
>
> The "⚠ Paper mismatch" notes below record what the repo code does. They are not decisions to change the paper.
>
> **File paths below use the repo layout from before the reorganization.** The table "Old → new locations" in the root `README.md` maps each one to its current place.

Source: branch `organized` of `aspect_extraction_and_classification`. File references are to that branch.
"⚠ Paper mismatch" marks places where the code does something different from what the paper says. Each one
needs either a paper fix or a team decision.

## Architecture

### Smart Router — classifier, features, categories
- **Classifier:** `LogisticRegression(class_weight="balanced", max_iter=1000)`, chosen over a DecisionTree (max_depth=20) and a soft-voting LR+DT ensemble by 5-fold stratified CV (`ROUTING_AND_ASPECT_CLASSIFICATION/FINALWINNERYAY.py`).
- **Features:** character TF-IDF, `analyzer="char_wb"`, n-grams 3–5, `max_features=10000`.
- **Classes are balanced by downsampling** the majority class (`random_state=42`).
- ⚠ **Only two categories: MSA vs DIALECT.** English vs Arabic is *not* decided by the router. In `whole_pipeline/whole_pipeline.py` it is a regex check for Arabic letters (`[؀-ۿ]`). `Dashboard/pipeline.py` uses `lingua` and the latency benchmark uses fastText `lid.176`. The reviewer's "three target categories" do not exist in the code.
- ⚠ **88.6% is training-fit accuracy.** The final model is refit on all data and scored on the same data, and Fig. 2 is that training confusion matrix. CV accuracy (83.9%) is the only held-out figure, and it is leaky because TF-IDF is fit on all rows before CV. **A rerun is needed for a real test accuracy.**
- ⚠ Fig. 2 sums to 4,720 rows (2,360 per class after balancing), not the 5,360 stated in §4.2.

### English pipeline — TF-IDF, SVM, cleaning
- **Cleaning for aspect extraction** (`english/normalizer_en.py`): curly quotes become ASCII, every character outside `[A-Za-z0-9\s,.\-!?']` is replaced by a space, and spaces are collapsed. **There is no lowercasing at this step.** Tokenization uses the regex `\w+|[^\w\s]`.
- **Cleaning for sentiment classification:** strip, lowercase, remove HTML tags, collapse whitespace.
- ⚠ **spaCy is not used** in the deployed English pipeline. It appears only in `english2/` (an unused CRF experiment), so Fig. 1, §3.3 and §6.2.1 overstate its role.
- ⚠ **Extraction does not use TF-IDF.** It is a `LinearSVC` on one-hot handcrafted token features (`DictVectorizer`), tuned by `GridSearchCV`:
  - grid: C ∈ {0.01, 0.1, 1, 2, 5}, class_weight ∈ {None, balanced}, loss ∈ {hinge, squared_hinge}, max_iter ∈ {2000, 3000, 5000};
  - 3-fold CV, scored by macro-F1.
- **TF-IDF is used only for aspect sentiment** (see "Sentiment classification" below).

### Arabic pipeline — LLM, normalization, MSA bypass
- **LLM:** the code disagrees with itself.
  - `Dashboard/whole_pipeline.py` uses **`qwen-plus-2025-04-28`** (Alibaba DashScope OpenAI-compatible API, `temperature=0`, `max_tokens=3000`).
  - `whole_pipeline/whole_pipeline.py` and `dialect_to_msa.py` use **`gpt-4.1`**.
  - Fig. 1 says Llama 3.
  - ⚠ **The team must state which model produced the reported results.**
- **How normalization works:** one LLM call per review. The system prompt is in Arabic; the full rules are in `whole_pipeline.py` lines 75-91. It says:
  - rewrite in MSA only;
  - keep the exact meaning, with no additions or omissions;
  - do not substitute synonyms;
  - keep brand names;
  - keep sentiment and tone;
  - remove letter elongation;
  - output the rewritten text only.
- **MSA reviews bypass the LLM.** If the router says MSA, the text goes straight to the Arabic models (`if variant == "DIALECT": convert … else: msa_text = text`).
- ⚠ **CAMeL Tools is not used anywhere in the code.** Orthographic normalization is a hand-written regex function (`normalize_arabic` in `Hamzah's_Part/normalizer.py` and its copies). It does the following:
  - removes diacritics (U+0617–061A, U+064B–0652);
  - removes tatweel;
  - maps أ إ آ → ا, ة → ه and ى → ي;
  - replaces every symbol except word characters, whitespace and ، ؟ . with a space;
  - collapses whitespace.
- ⚠ §5.1 also claims steps the extraction and sentiment pipelines never perform:
  - Arabic-Indic digit conversion;
  - stop-word removal;
  - rare-term filtering;
  - elongation reduction. Only the polarity notebook does this, with `(.)\1+ → \1`, which also collapses legitimate double letters.

### Aspect extraction — features, prediction, spans
- **Features per token** (`Hamzah's_Part/features.py`, `english/features_en.py`): token, lowercased token, `isdigit`, prefix-1/2, suffix-1/2, previous token, next token.
- ⚠ **The context window is ±1 token, not ±2** as §5.2 says. ⚠ There are also **no POS tags, no capitalization or punctuation features, and no character TF-IDF** in the extractor; POS and ±2 exist only in the unused CRF code.
- **Prediction:** each token is classified independently by `LinearSVC` (no sequence model). One-hot encoding is done with `DictVectorizer`.
- **Turning labels into spans** (`whole_pipeline/arabic_pipeline.py: extract_aspects`):
  - `B-ASP` opens a span;
  - following `I-ASP` tokens extend it;
  - `O` closes it;
  - an `I-ASP` with no open span is dropped.
- ⚠ **The deployed Arabic extractor is not the one evaluated in the paper.**
  - The pipeline loads `arabic_svm_grid.pkl`, produced by `training_svm_with_gridsearch.py` from the *old, buggy* `arabic_ate_features.json`.
  - The paper reports `arabic_svm_ate.pkl`, trained on the *fixed* `arabic_ate_features_fixed.json`.
- ⚠ **Span bugs in the English extractor:**
  - `english/prepare_dataset_en.py` assumes one space between every token when mapping characters to tokens, so offsets drift and aspects are mislabeled. `extras/biotagger.py` fixed this for Arabic only.
  - `whole_pipeline/english_pipeline.py: extract_aspects` computes span ends wrongly: it uses the previous token or the last token of the sentence.

### Sentiment classification — context and stacking
- **Context:** ±5 whitespace tokens around the aspect, with `ASP_START` / `ASP_END` marker tokens inserted around the aspect span, then lowercased.
- **Vectorizer** (the same for both languages): `FeatureUnion` of
  - word TF-IDF: n-grams 1–3, `max_features=20000`, `sublinear_tf`, `token_pattern=\w{1,}`;
  - `char_wb` TF-IDF: n-grams 3–5, `max_features=30000`, `sublinear_tf`.

  It is fit on the training split only.
- ⚠ §6.7.1 says English uses word n-grams only and Arabic uses word+char. **The code uses word+char for both.**
- **Stacking** (`get_stacking_model`):
  - base models:
    - `LinearSVC(C=0.5, class_weight=balanced)`;
    - `LogisticRegression(C=1, solver=liblinear, class_weight=balanced)`;
    - `SGDClassifier(loss=modified_huber, alpha=1e-4, class_weight=balanced)`;
    - `RidgeClassifier(class_weight=balanced)`;
  - meta-learner: `LogisticRegression(class_weight=balanced)`;
  - `StackingClassifier(cv=5)`, so **the meta-learner is trained on 5-fold out-of-fold predictions** ✓ (this answers the reviewer).
- **SMOTE** (`k_neighbors=5`) runs inside an `imblearn` pipeline on the training-split TF-IDF vectors, before stacking. Test data is never resampled ✓.
  - Caveat: SMOTE runs *before* stacking's internal CV, so synthetic points can be built from neighbors that later fall in other inner folds. This affects only the meta-learner's training signal, not test integrity.
- ⚠ Train/serve mismatch: the English inference code (`whole_pipeline/english_pipeline.py`) does **not** lowercase the window before classifying, but training did.

### General-polarity fallback
- ⚠ **It is not a fallback in the code.** `analyze_sentence_global` runs the general polarity model on **every** review, regardless of whether aspects were found. The result is returned as a separate field, `overall_polarity`, next to the aspect list in `absa`. That separate field is how its predictions are told apart from aspect-level ones.
- **English model:** `LinearSVC` on word TF-IDF (unigrams, English stop words, `max_features=50000`).
- **Arabic model:** `LogisticRegression(C=2)` on word TF-IDF (1–2-grams, `max_features=50000`).

### Post-processing / Qwen summarization
- ⚠ **There is no aggregation-to-LLM or summarization code anywhere in the repo.** It appears only as a bullet in the dashboard's description text. The Streamlit app's analyzer and dataset pages are stubs.
- **There are no "fields passed to Qwen" to report.** The options are to implement the step and evaluate it, or to remove it and the hallucination-reduction claim from the paper.

### Reproducibility settings
- **Seed:** `random_state=42` everywhere.
- **Splits:** 80/20 stratified everywhere, except English polarity, which uses the dataset's own split.
- Hyperparameters are listed above.

## Datasets

- **Router data (5,360):** labels in the code are only MSA/DIALECT, so English is not part of it. The LLM labelling script and `dataset.csv` are **not in the repo**; we need them from the team to report the distribution and the annotation method.
- **English ABSA, four labels:** `neutral` and `conflict` are **dropped** before binary training (`AspClassEnglish.py`).
  - Measured counts: 18,178 binary samples, **9,175 negative / 9,003 positive**. ⚠ The paper swaps these.
  - ⚠ The data is ~5.3k Amazon review sentences labelled by **GPT-4.1** (`extras/label.py`) plus the 3,030 SemEval-2014 laptop sentences, filtered to English with fastText. It is **not** "SemEval-2014 only" as §4.3 says.
- **Arabic ABSA, term vs category:** the XML has `<Opinion target=… from=… to=…>`, so **term-level extraction is supported** ✓.
  - **BIO derivation** (`extras/biotagger.py`):
    - keep positive/negative opinions only;
    - normalize the sentence and the target;
    - find every string occurrence of the target;
    - tokens fully inside an occurrence get B-ASP, then I-ASP;
    - `target="NULL"` opinions produce no tags.
  - ⚠ For sentiment classification, NULL targets (889 rows) are kept, with a window anchored at the sentence start. They are the top "aspect" in Table 9.
  - ⚠ Table 9's English glosses are wrong. The rows are الفندق "the hotel", فندق "hotel", الغرف "the rooms", الخدمة "the service". They are not hotel / room / service / meal.
- **Polarity splits:**
  - English: the dataset's own split after the fastText English filter, giving **3,580,535 train / 397,826 test**.
  - Arabic: 66,666 → stratified 80/20, giving **53,332 train / 13,334 test**. ⚠ The paper says the model was trained on the full 66,666.
- **Leakage found in the code:**
  - Arabic polarity: TF-IDF and GridSearch are fit on all 66,666 rows, including the later test rows.
  - Router: TF-IDF is fit on all rows before CV, and the final score is training fit.
  - Arabic and English extraction: the `DictVectorizer` is fit on all tokens, and the 80/20 split is **per token**, so a sentence's tokens land in both train and test.
  - Aspect sentiment and English polarity: clean ✓.
  - Router data vs ABSA/polarity data overlap: **needs `dataset.csv` to check.**

## Methodology / Results items answered from code

- **Which models use SMOTE:** only the two aspect-sentiment stacking classifiers (Arabic and English). ⚠ The BIO taggers do **not** use SMOTE; they use `class_weight="balanced"`. So the reviewer's "SMOTE on sequence-tagged data" concern is resolved by fixing the text in §5.3.
- **Why the Arabic macro-F1 values differ (0.7163 / 0.7104 / 0.7185):** they come from three scripts.
  - `evaluate_svm_split.py`: held-out split, `max_iter=1000`, tokens grouped by sentence string.
  - `train_svm_ate.py`: 5-fold CV mean.
  - `ablation_normalization.py`: `max_iter=5000`, tokens grouped per sentence occurrence.

  All three were reproduced exactly.
- **Latency — is the LLM included?** **No.** `extras/benchmark_latency.py` has no LLM stage.
  - ⚠ The script's English path has no extraction stage, but Table 15 reports one.
  - ⚠ Its Arabic classification stage classifies the full sentence, not the aspect window.
  - Hardware is not recorded.

  **A rerun is needed.**
