---
license: cc-by-4.0
language:
- sv
tags:
- text-classification
- cefr
- swedish
- sentence-transformers
- scikit-learn
pipeline_tag: text-classification
---

# Swedish CEFR Linear SVM Classifier

This repository contains the trained classifier artifacts for the Swedish CEFR
Text Classifier project.

The model predicts approximate CEFR reading levels (`A1` to `C2`) for Swedish
texts. Texts are embedded with
[`nicher92/saga-embed_v1`](https://huggingface.co/nicher92/saga-embed_v1), then
classified with the best-performing classical classifier from the project
evaluation.

## Artifacts

- `classifier.joblib`: trained Linear SVM classifier
- `label_encoder.joblib`: scikit-learn label encoder for CEFR labels
- `evaluation.csv`: held-out test-set classifier comparison
- `metadata.json`: task, label, embedding, and artifact metadata

## Evaluation

The classifier was evaluated on a stratified 25% test split from a 600-example
custom Swedish CEFR dataset.

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Linear SVM | 0.780 | 0.777 |
| Random Forest | 0.760 | 0.752 |
| Logistic Regression | 0.747 | 0.743 |
| KNN | 0.700 | 0.683 |

## Limitations

This is an educational classifier for approximate reading-level estimation, not
an official CEFR assessment. Predictions are most useful as a rough signal and
should be interpreted together with the probability scores shown in the demo.
