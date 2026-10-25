# Archive

These are earlier experiments that were superseded and are **not used for any number in the paper**. They are kept for reference and history. Their file paths point to the old repo layout, so they will not run as-is.

| Folder | What it was | Superseded by |
|---|---|---|
| `early_aspect_sentiment_v1/` | First aspect-sentiment attempt (`absa/`): 3-class, TF-IDF + LR/SVM/KNN/DT voting, FastText+SVM | `src/4_aspect_sentiment/` (binary, stacking) |
| `early_aspect_sentiment_v2/` | Cleaner leakage-safe rewrite of the above (`absa2/`), still 3-class (~71%) | `src/4_aspect_sentiment/` |
| `english_crf_tagger/` | CRF aspect tagger with POS and ±2-token features (`english2/`) | `src/3_aspect_extraction/english/` (SVM) |
| `bert_experiments/` | BERT-tokenized BIO conversion and a frozen-BERT-embedding classifier | not pursued (the paper is classical-ML only) |
| `arabic_bio_v1_buggy/` | First Arabic BIO preparation. Its offset bug dropped I-ASP labels | `src/3_aspect_extraction/arabic/prepare_bio_dataset.py` (restores 1,515 I-ASP labels) |
| `misc/` | Small one-off scripts (`training.py`, laptop XML → JSONL converter) | n/a |
| `not_part_of_project/` | Files committed by mistake: an unrelated IoT web project (`rdb_project/`), VHDL notes and a course scratch notebook | **can be deleted** |
