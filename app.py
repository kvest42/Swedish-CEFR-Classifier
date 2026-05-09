from pathlib import Path
import string

import gradio as gr
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from swedish_cefr_classifier import (
    DEFAULT_MODEL,
    LABEL_COLUMN,
    TEXT_COLUMN,
    classifier_candidates,
    create_embeddings,
    likely_matches,
    load_csv_dataset,
    load_embedding_model,
)


DATA_PATH = Path('swedish_cefr_dataset.tsv')
EVALUATION_PATH = Path('trained_model/evaluation.csv')
TEST_SIZE = 0.25
MATCH_THRESHOLD = 0.10
TOP_K = 6
TRAIN_EXAMPLES_PER_LABEL = 20
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation + '–—””“’‘…')
THEME = gr.themes.Soft(
    primary_hue='pink',
    secondary_hue='slate',
    neutral_hue='slate',
    radius_size='sm',
    text_size='md',
).set(
    body_background_fill='#09090f',
    body_background_fill_dark='#09090f',
    body_text_color='#f5f5f7',
    body_text_color_dark='#f5f5f7',
    block_background_fill='#14141f',
    block_background_fill_dark='#14141f',
    block_border_color='#2f2f46',
    block_border_color_dark='#2f2f46',
    block_label_text_color='#f9a8d4',
    block_label_text_color_dark='#f9a8d4',
    button_primary_background_fill='#ec4899',
    button_primary_background_fill_dark='#ec4899',
    button_primary_background_fill_hover='#f472b6',
    button_primary_background_fill_hover_dark='#f472b6',
    button_primary_text_color='#fff7fb',
    button_primary_text_color_dark='#fff7fb',
    input_background_fill='#0f0f18',
    input_background_fill_dark='#0f0f18',
    input_border_color='#3b3b55',
    input_border_color_dark='#3b3b55',
)

CSS = '''
html,
body,
.gradio-container,
.main,
.app,
.wrap,
.contain,
#root,
[data-testid="block-wrapper"] {
    background: #09090f !important;
    color: #f5f5f7 !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto;
    min-height: 100vh;
}

#hero {
    padding: 28px 0 10px;
}

#hero h1 {
    font-size: 38px;
    line-height: 1.08;
    margin-bottom: 10px;
}

#hero p {
    color: #c7c7d1;
    font-size: 16px;
    max-width: 760px;
}

#hero h1,
label,
.prose h1,
.prose h2,
.prose p,
.prose li,
.prose span {
    color: #f5f5f7 !important;
}

.metric-card {
    background: #14141f !important;
    border: 1px solid #2f2f46 !important;
    border-radius: 8px;
    padding: 14px 16px;
}

.panel {
    background: #14141f !important;
    border: 1px solid #2f2f46 !important;
    border-radius: 8px;
}

textarea,
input,
.wrap input,
.wrap textarea,
.block input,
.block textarea {
    background: #0f0f18 !important;
    border-color: #3b3b55 !important;
    color: #f5f5f7 !important;
    font-size: 16px !important;
}

button.primary,
#classify-button {
    background: #ec4899 !important;
    border-color: #f472b6 !important;
    color: #fff7fb !important;
}

button.primary:hover,
#classify-button:hover {
    background: #f472b6 !important;
}

#classify-button {
    min-height: 44px;
    font-weight: 700;
}

#summary textarea {
    font-weight: 700;
    color: #f9a8d4 !important;
}

.label-wrap,
.output-class,
.table-wrap,
.dataframe,
.tabitem,
.accordion,
.examples,
.dataset,
.block {
    background: #14141f !important;
    border-color: #2f2f46 !important;
    color: #f5f5f7 !important;
}

.label-wrap span,
.output-class span,
.examples span,
.accordion span {
    color: #f5f5f7 !important;
}

.bar {
    background: #ec4899 !important;
}

table,
thead,
tbody,
tr,
td,
th {
    background: #0f0f18 !important;
    color: #f5f5f7 !important;
    border-color: #2f2f46 !important;
}

footer {
    display: none !important;
}
'''


def train_demo():
    full_data = load_csv_dataset(DATA_PATH)
    data = (
        full_data.groupby(LABEL_COLUMN, group_keys=False)
        .sample(n=TRAIN_EXAMPLES_PER_LABEL, random_state=42)
        .reset_index(drop=True)
    )
    print(f'Loaded {len(full_data)} examples; training demo on {len(data)} balanced examples.')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[LABEL_COLUMN].tolist())

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        data[TEXT_COLUMN].tolist(),
        y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y,
    )

    print(f'Loading embedding model: {DEFAULT_MODEL}')
    embedding_model = load_embedding_model(DEFAULT_MODEL)
    print('Creating training embeddings.')
    x_train = create_embeddings(embedding_model, x_train_text)
    print('Creating test embeddings.')
    x_test = create_embeddings(embedding_model, x_test_text)

    classifiers = classifier_candidates()
    rows = []
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        rows.append(
            {
                'classifier': name,
                'accuracy': round(accuracy_score(y_test, predictions), 3),
                'macro_f1': round(f1_score(y_test, predictions, average='macro'), 3),
            }
        )

    results = pd.DataFrame(rows).sort_values(['macro_f1', 'accuracy'], ascending=False)
    best_name = results.iloc[0]['classifier']
    best_classifier = classifiers[best_name]
    print(f'Best classifier: {best_name}')
    return full_data, data, label_encoder, embedding_model, best_classifier, best_name, results


def load_official_evaluation():
    if EVALUATION_PATH.exists():
        return pd.read_csv(EVALUATION_PATH).round(3)
    return pd.DataFrame(
        [
            {'classifier': 'Linear SVM', 'accuracy': 0.867, 'macro_f1': 0.866},
            {'classifier': 'Logistic Regression', 'accuracy': 0.850, 'macro_f1': 0.848},
            {'classifier': 'Random Forest', 'accuracy': 0.808, 'macro_f1': 0.805},
            {'classifier': 'KNN', 'accuracy': 0.708, 'macro_f1': 0.704},
        ]
    )


FULL_DATA, TRAIN_DATA, LABEL_ENCODER, EMBEDDING_MODEL, CLASSIFIER, BEST_NAME, RESULTS = train_demo()
OFFICIAL_RESULTS = load_official_evaluation()


def classify_text(text: str):
    text = text.strip()
    if not text:
        return 'Write a Swedish text first.', {}, ''

    normalized_text = text.translate(PUNCTUATION_TABLE)
    embedding = create_embeddings(EMBEDDING_MODEL, [normalized_text])
    probabilities = CLASSIFIER.predict_proba(embedding)[0]
    labels = LABEL_ENCODER.classes_.tolist()
    matches = likely_matches(probabilities, labels, MATCH_THRESHOLD, TOP_K)
    best_label = matches[0][0]
    best_score = matches[0][1]
    summary = f'{best_label} ({best_score:.0%})'
    scores = {label: float(score) for label, score in matches}
    details = '\n'.join(f'{label}: {score:.1%}' for label, score in matches)
    return summary, scores, details


with gr.Blocks(title='Swedish CEFR Classifier', theme=THEME, css=CSS) as demo:
    gr.Markdown(
        '''
        # Swedish CEFR Text Classifier

        Estimate the CEFR reading level of a Swedish text with sentence embeddings and classical classifiers.
        The demo uses `nicher92/saga-embed_v1` and a custom dataset labeled from A1 to C2.
        ''',
        elem_id='hero',
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes='metric-card'):
            gr.Number(value=len(FULL_DATA), label='Dataset examples', interactive=False)
        with gr.Column(scale=1, elem_classes='metric-card'):
            gr.Number(value=len(TRAIN_DATA), label='Demo training examples', interactive=False)
        with gr.Column(scale=1, elem_classes='metric-card'):
            gr.Textbox(value=BEST_NAME, label='Best classifier', interactive=False)

    with gr.Row():
        with gr.Column(scale=3, elem_classes='panel'):
            text_input = gr.Textbox(
                label='Swedish text',
                lines=9,
                value='Jag går till skolan varje morgon.',
                placeholder='Paste a Swedish text here...',
            )
            button = gr.Button('Classify text', variant='primary', elem_id='classify-button')

        with gr.Column(scale=2, elem_classes='panel'):
            summary_output = gr.Textbox(label='Best CEFR match', elem_id='summary')
            label_output = gr.Label(label='Probability distribution', num_top_classes=6)
            details_output = gr.Textbox(label='Scores', lines=6)

    button.click(classify_text, inputs=text_input, outputs=[summary_output, label_output, details_output])

    gr.Examples(
        examples=[
            ['Jag vill ha kaffe och en smörgås'],
            ['Kommunen behöver prioritera resurser när flera projekt konkurrerar om pengar'],
            ['Åtgärdens proportionalitet måste prövas i relation till dokumenterade behov och alternativa handlingsvägar'],
        ],
        inputs=text_input,
        label='Try examples',
    )

    with gr.Accordion('Classifier evaluation', open=False):
        gr.Markdown(
            '''
            Official benchmark from `swedish_cefr_train.tsv` and `swedish_cefr_test.tsv`.
            The live demo trains on a smaller balanced sample so the Space starts quickly.
            '''
        )
        gr.Dataframe(value=OFFICIAL_RESULTS, label='Official held-out test set', interactive=False)


if __name__ == '__main__':
    demo.launch()
