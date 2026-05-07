# Swedish CEFR Text Classifier

This project trains an embedding-based classifier that predicts approximate CEFR reading level for Swedish text.

Labels:

- A1
- A2
- B1
- B2
- C1
- C2

The project uses a custom dataset, Swedish sentence embeddings from `nicher92/saga-embed_v1`, several scikit-learn classifiers, and a Gradio web demo for Hugging Face Spaces.

## Files

| path | purpose |
| --- | --- |
| `swedish_cefr_dataset.tsv` | Custom Swedish CEFR-style text classification dataset |
| `swedish_cefr_classifier.py` | Training, evaluation, and command-line prediction script |
| `huggingface_upload/` | Files prepared for the Hugging Face dataset repo |
| `huggingface_space/` | Files prepared for the Hugging Face Gradio demo |

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install pandas scikit-learn sentence-transformers datasets gradio
```

## Train and Evaluate

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv
```

The script trains and compares:

- Logistic Regression
- Linear SVM
- Random Forest
- KNN

It reports accuracy, macro F1, a classification report, and a confusion matrix.

## Predict a Text

```bash
python3 swedish_cefr_classifier.py --data swedish_cefr_dataset.tsv --text 'Jag går till skolan varje morgon.'
```

## Run the Web Demo Locally

```bash
cd huggingface_space
python3 app.py
```

## Upload Dataset to Hugging Face

```bash
information-retrieval/.venv/bin/hf upload kvest/swedish-cefr-text-complexity information-retrieval/huggingface_upload . --repo-type dataset
```

## Upload Demo to Hugging Face Spaces

Create a new Space on Hugging Face with SDK `Gradio`, then upload:

```bash
information-retrieval/.venv/bin/hf upload kvest/swedish-cefr-text-classifier information-retrieval/huggingface_space . --repo-type space
```

## Submission Links

Fill these into your own report:

- GitHub repo: TODO
- Hugging Face dataset: `https://huggingface.co/datasets/kvest/swedish-cefr-text-complexity`
- Hugging Face demo: `https://huggingface.co/spaces/kvest/swedish-cefr-text-classifier`

## Limitations

The dataset labels are approximate CEFR-style labels. They are suitable for an educational machine learning project, but they are not official CEFR test results and should not be used for high-stakes language assessment.
