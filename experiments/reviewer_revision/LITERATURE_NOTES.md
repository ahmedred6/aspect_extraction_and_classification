# Published results for the SOTA comparison table (D10)

Every number below was read directly from the paper's PDF (in `Downloads/AI_Semantic_Classification/literature/`, not in the repo). The table or page is cited for each one. Nothing comes from memory. Read on 2026-11-14.

**Comparability warning.** All published results below use the **official SemEval test sets**. Our results use random 80/20 splits (or 5-fold CV) of the **training** data. The comparison paragraph must say this explicitly.

## English: SemEval-2014 Task 4 (laptops, restaurants)

| Ref | System | Aspect term extraction F1 (exact span) | Aspect sentiment | Classes | Source |
|---|---|---|---|---|---|
| [6] | Official baseline | Lap 35.64 / Rest 47.15 | Acc: Lap 51.37 / Rest 64.28 | 4 (pos/neg/neu/conflict) | Pontiki 2014, Tables 4 & 6 (pp. 32, 34) |
| [6] | Best participant | Lap 74.55 (IHS_RD, CRF) / Rest 84.01 (DLIREC, unconstrained) | Acc: Lap 70.48 (DCU, NRC-Canada, SVM) / Rest 80.95 (DCU) | 4 | same |
| [11] | ATAE-LSTM | n/a | Rest acc 77.2 (3-class) / 90.9 (pos/neg); Lap 68.7 / 87.6 | 3 and 2 | Wang 2016, Tables 3–4 (p. 612) |
| [13] | ASGCN-DG | n/a | Lap14 acc 75.55 / macro-F1 71.05; Rest14 80.77 / 72.02 | 3 (avg of 3 runs) | Zhang 2019, Table 2 (p. 4573) |
| [15] | BERT-PT (BERT-base) | Lap (SemEval14) 84.26; Rest (SemEval16) 77.97 | Lap acc 78.07 / MF1 75.08; Rest 84.95 / 76.96 | 3 (conflict dropped; avg of 9 runs) | Xu 2019, Tables 5–6 (p. 2332) |
| [16] | BERT-pair (Sun 2019) | n/a (category-level task) | category-level / SentiHood | n/a | **Not comparable** (aspect categories, not terms) |

**Notes**
- ATAE-LSTM's **binary (pos/neg) accuracy of 90.9% (Rest) and 87.6% (Lap)** is the closest setting to our binary task.
- BERT-PT ran on a GPU (11 GB) with FP16; BERT-base has 110M parameters.

## Arabic: SemEval-2016 Task 5, Arabic hotels (AR/HOTE)

| Ref | System | Opinion target extraction (Slot 2) F1 | Polarity (Slot 3) acc | Source |
|---|---|---|---|---|
| [19] | Official baseline | 30.978 | 76.421 | Pontiki 2016, Table 4 (p. 26) |
| [19] | Best participant | **no participant submitted Arabic Slot 2** | 82.719 (INSIG., constrained) | same |
| [7] | Al-Smadi 2019, Bi-LSTM-CRF (fastText char-level) / AB-LSTM-PC | **69.98** (word2vec variant: 66.32) | **82.6** | Al-Smadi 2019, Table 2 (p. 8); PDF provided by the user |
| [7] | (reported there) INSIGHT-1 CNN / IIT-TUDA SVM + lexicon / IIT-TUDA SVM | n/a | 82.7 / 81.72 / 78.68 | same |

- **Setup of [7]:** a 70/10/20 train/val/test split of the Arabic hotels reviews (§5.1), not the official test set. That makes it **closer to our random-split protocol** than the SemEval results are.
- **Metric of [7]:** F1 is not defined explicitly. They compare against the SemEval baseline of 30.9, so it is most likely SemEval-style exact-span F1.
- **Classes of [7]:** the number of polarity classes in T2 is not stated (SemEval Arabic has positive/negative/neutral). Ours is binary, so do not claim superiority on accuracy alone.
- ⚠ **Citation mismatch:** our §2.4 says Al-Smadi et al. built "SVM and CRF based systems". Paper [7] is **LSTM-based** (Bi-LSTM-CRF plus aspect-based attention LSTM). The SVM system on this dataset is IIT-TUDA, cited inside [7]. Fix the §2.4 sentence.

## Arabic general polarity (context for Tables 13–14)

| Ref | System | Dataset | Acc | Source |
|---|---|---|---|---|
| [23] | AraBERT v0.1 / v1 | HARD | 96.2 / 96.1 | Antoun 2020, Table 1 (p. 13) |
| [23] | AraBERT v0.1 / v1 | LABR (unbalanced, 2-class) | 85.9 / 86.7 | same |

- **Not directly comparable.** Our Arabic polarity data is the Arabic 100K Reviews corpus, not HARD or LABR.

## Computational cost

| Ref | Model | Params | CPU inference (full STS-B dev pass, batch 1) | Source |
|---|---|---|---|---|
| [17] | BERT-base | 110M | 668 s | Sanh 2019, Table 3 (p. 3) |
| [17] | DistilBERT | 66M | 410 s (60% faster, 97% of BERT's GLUE score) | same; Intel Xeon E5-2690 v3 |

- These figures are a **full-dataset pass on a different CPU**, so they cannot be converted to ms/review comparable with our Table 15. Report parameter counts, not speed comparisons (consistent with D5).
