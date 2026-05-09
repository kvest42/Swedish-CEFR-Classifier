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

The classifier was evaluated on a separate 120-example held-out test dataset
after training on 600 Swedish CEFR examples. The explicit train and test files
were shuffled reproducibly with `random_state=1004`.

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Linear SVM | 0.867 | 0.866 |
| Logistic Regression | 0.850 | 0.848 |
| Random Forest | 0.808 | 0.805 |
| KNN | 0.708 | 0.704 |

## Limitations

This is an educational classifier for approximate reading-level estimation, not
an official CEFR assessment. Predictions are most useful as a rough signal and
should be interpreted together with the probability scores shown in the demo.
