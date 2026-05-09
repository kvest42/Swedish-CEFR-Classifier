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

The exported classifier is a **Linear SVM** trained on normalized sentence
embeddings. The embedding model handles the Swedish text representation, while
the Linear SVM learns decision boundaries between the six CEFR levels in that
embedding space.

## Artifacts

- `classifier.joblib`: trained Linear SVM classifier
- `label_encoder.joblib`: scikit-learn label encoder for CEFR labels
- `evaluation.csv`: held-out test-set classifier comparison
- `metadata.json`: task, label, embedding, and artifact metadata

## Evaluation

The classifier was evaluated on a separate 120-example held-out test dataset
after training on 600 Swedish CEFR examples. The explicit train and test files
were shuffled reproducibly with `random_state=1004`.

Four scikit-learn classifiers were compared on the same embeddings:

- **Linear SVM**: margin-based linear classifier and final selected model.
- **Logistic Regression**: simple linear probabilistic baseline.
- **Random Forest**: non-linear tree ensemble baseline.
- **KNN**: nearest-neighbor similarity baseline.

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Linear SVM | 0.867 | 0.866 |
| Logistic Regression | 0.850 | 0.848 |
| Random Forest | 0.808 | 0.805 |
| KNN | 0.708 | 0.704 |

The Linear SVM was selected because it achieved the highest accuracy and macro
F1. Macro F1 is especially useful here because each CEFR class should matter
equally, not only the most frequent class. The train and test sets are balanced,
with 100 training examples and 20 test examples per label.

The strong performance of Linear SVM and Logistic Regression suggests that the
embedding model already organizes the texts in a way that makes the reading
levels mostly linearly separable. Random Forest did capture some useful
non-linear patterns, but it did not improve over the linear classifiers. KNN
was weaker, indicating that nearest-neighbor similarity alone is not as reliable
for this six-level classification problem.

## Intended Use

This model is intended for approximate Swedish text-complexity classification
and educational experimentation with embeddings plus classical classifiers. A
typical workflow is:

1. Embed Swedish input text with `nicher92/saga-embed_v1`.
2. Load `classifier.joblib` and `label_encoder.joblib`.
3. Predict the CEFR label and inspect probability scores.

The model should be used as a rough reading-level signal rather than an official
CEFR placement tool.

## Limitations

This is an educational classifier for approximate reading-level estimation, not
an official CEFR assessment. Predictions are most useful as a rough signal and
should be interpreted together with the probability scores shown in the demo.
