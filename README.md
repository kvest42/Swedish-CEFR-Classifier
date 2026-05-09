---
title: Swedish CEFR Text Classifier
emoji: 📘
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
python_version: 3.11
pinned: false
license: cc-by-4.0
---

# Swedish CEFR Text Classifier

This Space estimates the CEFR reading level of Swedish text. Paste a sentence or
paragraph into the demo and it returns the closest level from A1 to C2, together
with probability scores for the most likely levels.

The project was built for an information retrieval assignment and demonstrates a
simple, interpretable pipeline: Swedish sentence embeddings plus classical
machine-learning classifiers.

## Project Links

- Demo Space: <https://huggingface.co/spaces/kvest/swedish-cefr-text-classifier>
- Dataset: <https://huggingface.co/datasets/kvest/swedish-cefr-text-complexity>
- Trained model artifacts: <https://huggingface.co/kvest/swedish-cefr-linear-svm>

## What the demo does

- Embeds Swedish text with [`nicher92/saga-embed_v1`](https://huggingface.co/nicher92/saga-embed_v1).
- Trains several classifiers on a local Swedish CEFR dataset.
- Selects the best classifier by macro F1 and accuracy on a held-out test split.
- Shows the predicted CEFR level and a probability distribution over likely levels.

The current Space trains on a balanced sample from `swedish_cefr_dataset.tsv`.
The repository also includes explicit train/test dataset files:
`swedish_cefr_train.tsv` with 600 examples and `swedish_cefr_test.tsv` with
120 held-out examples. Each CEFR label has 100 training examples and 20 test
examples.

## Classifiers

The app compares four classical classifiers:

- Logistic Regression
- Linear SVM
- Random Forest
- KNN

The best-performing classifier from the demo training run is used for live
predictions in the Gradio interface.

## Evaluation

The classifiers were evaluated on the explicit held-out test dataset
(`swedish_cefr_test.tsv`) after training on `swedish_cefr_train.tsv`. The
training dataset has 600 examples and the test dataset has 120 examples. Each
CEFR label has 100 training examples and 20 test examples. The best model was a
Linear SVM trained on normalized `nicher92/saga-embed_v1` sentence embeddings.

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Linear SVM | 0.867 | 0.866 |
| Logistic Regression | 0.850 | 0.848 |
| Random Forest | 0.783 | 0.777 |
| KNN | 0.708 | 0.704 |

The exported model repository contains the trained Linear SVM classifier,
label encoder, evaluation table, and metadata. The classifier artifact was
retrained on all 600 examples after model selection.

## CEFR Labels

The labels are approximate reading-level categories:

- `A1`: very simple everyday language
- `A2`: simple familiar language
- `B1`: clear general-purpose language
- `B2`: more detailed or abstract language
- `C1`: advanced language with complex structure
- `C2`: highly complex or specialized language

## Run Locally

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the Gradio app:

```bash
python3 app.py
```

You can also run the classifier script directly:

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv --text "Jag går till skolan varje morgon."
```

## Data Notes

The repository includes a small custom TSV file with Swedish example texts and
CEFR-style labels. The classifier script can also load the public
[`UppsalaNLP/swedish-text-complexity`](https://huggingface.co/datasets/UppsalaNLP/swedish-text-complexity)
dataset and derive approximate labels from LIX readability scores.

The published version of the custom project dataset is available at
[`kvest/swedish-cefr-text-complexity`](https://huggingface.co/datasets/kvest/swedish-cefr-text-complexity).
It includes the full dataset plus the explicit train and test TSV files used
for evaluation.

Example:

```bash
python3 swedish_cefr_classifier.py --huggingface --hf-dataset UppsalaNLP/swedish-text-complexity
```

To export the best classifier after evaluation:

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv --save-model-dir trained_model
```

## Limitations

This is an educational demo, not an official CEFR assessment. CEFR reading level
depends on vocabulary, grammar, topic familiarity, discourse structure, and the
reader's background knowledge. Short inputs can be especially uncertain, so the
probability distribution is often more informative than the single top label.
