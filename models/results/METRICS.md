# satquery -- quantitative evaluation

Higher is better for every row except `object count MAE`.
BLEU / METEOR / ROUGE / BERTScore are shown as percentages; CIDEr-D is scaled
x100, the convention in the LEVIR-CC change-captioning literature.

The base model is scored on its RAW prose (it emits no JSON), which is generous
to it: the caption metrics judge description quality alone. Its inability to
produce the schema shows up only in `valid schema JSON`.

READ THE SUBSETS, NOT JUST THE AGGREGATE. Half of this evaluation set is
unchanged pairs, and LEVIR-CC labels almost all of those with the same handful
of sentences ("the scene is the same as before"). A model that has learned to
recognise no-change scores near-perfect n-gram overlap on that half, which pulls
the `all rows` averages up. The `changed pairs only` block is the honest
difficulty: it is real description against five varied human captions, and it is
the number to quote when comparing against published LEVIR-CC results.

CIDEr document frequencies are computed over this evaluation set, so the
absolute value depends on set size -- compare the columns against each other,
not against a published number. On the `unchanged pairs only` subset CIDEr
collapses toward zero for EVERY model, base and tuned alike: LEVIR-CC gives
near-identical no-change captions to all of those pairs, so their n-grams carry
no inverse document frequency and earn no credit. That is CIDEr working as
designed (it rewards distinctive agreement), not a scoring failure -- read the
change-detection rows there instead.

METEOR backend: `builtin` | bootstrap resamples: 2000

### BI-TEMPORAL CHANGE CAPTIONING (LEVIR-CC)  --  all rows, n=120

| metric | base model | fine-tuned | delta | 95% CI (fine-tuned) |
|---|---|---|---|---|
| BLEU-1 | 17.85 | 76.68 | +58.83 | [72.33, 81.04] |
| BLEU-2 | 8.80 | 66.19 | +57.39 | [60.70, 71.71] |
| BLEU-3 | 3.35 | 57.58 | +54.22 | [51.29, 63.92] |
| BLEU-4 | 1.10 | 51.02 | +49.91 | [44.21, 57.83] |
| METEOR | 17.80 | 69.29 | +51.50 | [63.26, 74.98] |
| ROUGE-1 | 20.67 | 71.98 | +51.32 | [66.68, 77.18] |
| ROUGE-2 | 4.84 | 60.31 | +55.47 | [52.84, 67.50] |
| ROUGE-L | 19.84 | 69.34 | +49.50 | [63.57, 74.97] |
| CIDEr-D (x100) | 2.50 | 123.63 | +121.13 | [107.62, 138.95] |
| BERTScore F1 | 20.41 | 68.08 | +47.67 | [63.85, 71.98] |
| valid schema JSON (%) | 0.00 | 100.00 | +100.00 | -- |
| recall, changed (%) | 100.00 | 91.67 | -8.33 | -- |
| recall, unchanged (%) | 0.00 | 96.67 | +96.67 | -- |
| balanced accuracy (%) | 50.00 | 94.17 | +44.17 | -- |
| changed-class F1 (%) | 0.00 | 88.62 | +88.62 | -- |

### BI-TEMPORAL CHANGE CAPTIONING (LEVIR-CC)  --  changed pairs only, n=60

| metric | base model | fine-tuned | delta | 95% CI (fine-tuned) |
|---|---|---|---|---|
| BLEU-1 | 27.23 | 65.10 | +37.87 | [60.75, 69.69] |
| BLEU-2 | 14.95 | 48.58 | +33.63 | [43.64, 53.89] |
| BLEU-3 | 5.37 | 34.64 | +29.27 | [29.05, 40.15] |
| BLEU-4 | 1.94 | 24.63 | +22.69 | [18.67, 30.20] |
| METEOR | 18.70 | 41.50 | +22.80 | [37.33, 45.92] |
| ROUGE-1 | 25.15 | 46.34 | +21.19 | [43.31, 49.74] |
| ROUGE-2 | 7.90 | 23.95 | +16.05 | [20.09, 28.31] |
| ROUGE-L | 22.44 | 41.23 | +18.79 | [38.12, 44.59] |
| CIDEr-D (x100) | 2.86 | 41.29 | +38.43 | [28.57, 56.14] |
| BERTScore F1 | 26.54 | 49.79 | +23.25 | [46.52, 52.89] |
| valid schema JSON (%) | 0.00 | 100.00 | +100.00 | -- |
| recall, changed (%) | 100.00 | 91.67 | -8.33 | -- |
| recall, unchanged (%) | n/a | n/a | n/a | -- |
| balanced accuracy (%) | n/a | n/a | n/a | -- |
| changed-class F1 (%) | 0.00 | 88.62 | +88.62 | -- |

### BI-TEMPORAL CHANGE CAPTIONING (LEVIR-CC)  --  unchanged pairs only, n=60

| metric | base model | fine-tuned | delta | 95% CI (fine-tuned) |
|---|---|---|---|---|
| BLEU-1 | 10.49 | 96.25 | +85.76 | [90.64, 100.00] |
| BLEU-2 | 3.75 | 95.54 | +91.79 | [88.95, 100.00] |
| BLEU-3 | 1.67 | 95.18 | +93.51 | [88.12, 100.00] |
| BLEU-4 | 0.00 | 94.86 | +94.86 | [87.40, 100.00] |
| METEOR | 16.89 | 97.09 | +80.20 | [92.89, 99.85] |
| ROUGE-1 | 16.18 | 97.63 | +81.45 | [93.95, 100.00] |
| ROUGE-2 | 1.78 | 96.67 | +94.89 | [91.67, 100.00] |
| ROUGE-L | 17.24 | 97.46 | +80.21 | [93.63, 100.00] |
| CIDEr-D (x100) | 0.00 | 0.00 | +0.00 | [0.00, 0.00] |
| BERTScore F1 | 14.28 | 86.37 | +72.09 | [82.73, 88.74] |
| valid schema JSON (%) | 0.00 | 100.00 | +100.00 | -- |
| recall, changed (%) | n/a | n/a | n/a | -- |
| recall, unchanged (%) | 0.00 | 96.67 | +96.67 | -- |
| balanced accuracy (%) | n/a | n/a | n/a | -- |
| changed-class F1 (%) | n/a | n/a | n/a | -- |

### SINGLE-IMAGE DESCRIPTION (VRSBench)  --  all rows, n=60

| metric | base model | fine-tuned | delta | 95% CI (fine-tuned) |
|---|---|---|---|---|
| BLEU-1 | 14.68 | 44.51 | +29.83 | [40.52, 46.14] |
| BLEU-2 | 7.65 | 27.73 | +20.08 | [24.56, 29.52] |
| BLEU-3 | 3.87 | 17.95 | +14.08 | [15.27, 19.74] |
| BLEU-4 | 1.97 | 12.06 | +10.08 | [9.80, 13.75] |
| METEOR | 26.01 | 37.12 | +11.11 | [34.13, 39.93] |
| ROUGE-1 | 23.18 | 44.80 | +21.63 | [42.34, 47.15] |
| ROUGE-2 | 6.06 | 17.56 | +11.50 | [15.41, 19.65] |
| ROUGE-L | 17.27 | 33.65 | +16.38 | [31.61, 35.52] |
| CIDEr-D (x100) | 0.64 | 25.10 | +24.47 | [15.98, 35.72] |
| BERTScore F1 | 15.03 | 44.37 | +29.34 | [42.04, 46.60] |
| valid schema JSON (%) | 0.00 | 100.00 | +100.00 | -- |
| object count exact (%) | n/a | 55.00 | n/a | -- |
| object count MAE | n/a | 0.67 | n/a | -- |
| object-class F1, set (%) | 0.00 | 63.31 | +63.31 | -- |
| object-class F1, multiset (%) | 0.00 | 57.14 | +57.14 | -- |
