---
license: cc-by-4.0
language:
- sv
pretty_name: Swedish CEFR Text Complexity Dataset
task_categories:
- text-classification
tags:
- swedish
- cefr
- readability
- text-complexity
- education
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: swedish_cefr_dataset.tsv
---

# Swedish CEFR Text Complexity Dataset

This dataset contains short Swedish text examples labeled with approximate CEFR reading levels:

- A1
- A2
- B1
- B2
- C1
- C2

It was created for an information retrieval class project about training a classifier with sentence embeddings.
The intended task is to predict which CEFR level a Swedish text is closest to.

## Dataset Structure

The dataset is stored as a TSV file with two columns:

| column | description |
| --- | --- |
| `text` | Swedish text example |
| `label` | Approximate CEFR level |

The dataset contains 600 examples, balanced across six labels:

| label | examples |
| --- | ---: |
| A1 | 100 |
| A2 | 100 |
| B1 | 100 |
| B2 | 100 |
| C1 | 100 |
| C2 | 100 |

## Intended Use

This dataset is intended for small-scale educational experiments in Swedish text classification, readability analysis, and sentence embeddings.

Example project:

> Train a classifier that embeds Swedish texts with `nicher92/saga-embed_v1` and predicts the most likely CEFR reading level.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset('YOUR_USERNAME/swedish-cefr-text-complexity')
```

## Baseline Script

This repository also includes `swedish_cefr_classifier.py`, a simple baseline script using:

- `nicher92/saga-embed_v1`
- `sentence-transformers`
- `scikit-learn`
- logistic regression

Run locally:

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv
```

Classify a custom text:

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv --text 'Jag går till skolan varje morgon.'
```

## Limitations

The CEFR labels are approximate and were created as educational examples. They are not official CEFR assessment results and should not be used for high-stakes language testing.

The dataset is small, so model performance may be unstable. It is best used for coursework, prototyping, and demonstrating an embedding-based classification pipeline.

## License

This dataset is released under CC BY 4.0.
