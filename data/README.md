# Data

`raw/` holds the input datasets. `processed/` is where scripts write everything they generate; it is git-ignored and can always be regenerated.

## `raw/`

| File | What it is | Used by (paper) | Source |
|---|---|---|---|
| `semeval2014_laptops_train_v2.xml` | SemEval-2014 Task 4, Laptop train (3,045 sentences) | English aspect extraction (§6.2) | SemEval-2014 Task 4 |
| `semeval2014_restaurants_train_v2.xml` | SemEval-2014 Task 4, Restaurant train (3,041 sentences) | English aspect extraction: Table 3 uses laptop + restaurant | Hugging Face mirror `alexcadillon/SemEval2014Task4` |
| `semeval2016_arabic_hotels_train_v2.xml` | SemEval-2016 Task 5, Arabic hotels: 4,802 sentences, 10,509 opinions with target terms, offsets and categories | Arabic aspect extraction (§6.4–6.6, Tables 5–7) | SemEval-2016 Task 5 |
| `semeval2016_arabic_hotels_cleaned.xml` | Same 10,509 opinions, with category attributes and 12 opinion-less sentences removed (4,790 sentences) | Arabic aspect sentiment (§6.7, Tables 8–9) | derived from the above |
| `amazon_absa_gpt41_labeled.jsonl` | ~5.3k Amazon review sentences with aspect terms/polarities labelled by GPT-4.1, plus the 3,030 SemEval laptop sentences (8,325 rows) | English aspect sentiment (§6.7, Tables 8, 10), after `src/4_aspect_sentiment/filter_english_sentences.py` | built with `src/data_labeling/` (was `final_dataset.jsonl`) |
| `amazon_reviews_8000.txt` | 8,000 Amazon reviews, the input to the GPT-4.1 labelling | `src/data_labeling/label_english_absa_gpt41.py` | first 8,000 of Amazon Review Polarity train |
| `arabic_100k_reviews.tsv` | Arabic 100K Reviews (Positive/Negative/Mixed) | Arabic general polarity (§6.8, Tables 13–14) | Kaggle "Arabic 100k Reviews" |
| `laptop_prompt_test_sentences.txt` | First 100 laptop sentences, used to test labelling prompts | `src/data_labeling/` | from the laptop XML |
| `amazon_polarity/` ⚠ **download** | Amazon Review Polarity, 3.6M train / 400k test (1.7 GB, git-ignored) | English general polarity (§6.8, Tables 11–12) | see below |
| `router_dataset.csv` ⚠ **missing** | 5,360 LABR + HARD reviews labelled MSA/DIALECT by an LLM | Smart Router (§6.1) | ask the team |

### Downloading Amazon Review Polarity

```bash
curl -L -o amazon.tgz https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz
tar -xzf amazon.tgz
mv amazon_review_polarity_csv data/raw/amazon_polarity
```

Then run `python src/5_general_polarity/filter_english_amazon_reviews.py`. It keeps English reviews only: 3,580,535 train / 397,826 test, matching the paper.
