'''
Swedish CEFR level classification with sentence embeddings.

This is a starter project for an information retrieval assignment:
given a Swedish text, predict which CEFR reading level it best matches.

Real dataset:
https://huggingface.co/datasets/UppsalaNLP/swedish-text-complexity

The dataset contains Swedish texts with readability metrics such as LIX,
OVIX, average sentence length, average word length, and long-word percentage.

Install dependencies:
    pip install datasets pandas scikit-learn sentence-transformers

Run with a Hugging Face dataset:
    python3 swedish_cefr_classifier.py --huggingface --hf-dataset YOUR_USERNAME/swedish-cefr-text-complexity

Classify one or more texts after training:
    python3 swedish_cefr_classifier.py --huggingface --hf-dataset YOUR_USERNAME/swedish-cefr-text-complexity --text 'Beslutet gäller från och med nästa månad.'

CEFR labels:
    A1, A2, B1, B2, C1, C2
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
LIX_COLUMN = 'lix'
DEFAULT_MODEL = 'nicher92/saga-embed_v1'
HF_DATASET = 'UppsalaNLP/swedish-text-complexity'
RANDOM_STATE = 1004


def load_pandas():
    '''Import pandas with a helpful project-level install message.'''
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: pandas\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    return pd


def load_csv_dataset(path: Path | None) -> pd.DataFrame:
    '''Load a local text classification dataset or return a tiny demo dataset.'''
    pd = load_pandas()

    if path is None:
        return demo_dataset()

    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')

    separator = '\t' if path.suffix.lower() in {'.tsv', '.tab'} else ','
    data = pd.read_csv(path, sep=separator)

    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(data.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f'Dataset is missing required column(s): {missing}')

    return clean_dataset(data[[TEXT_COLUMN, LABEL_COLUMN]])


def load_huggingface_dataset(dataset_name: str, label_scheme: str) -> pd.DataFrame:
    '''Load Swedish text complexity data from Hugging Face.'''
    pd = load_pandas()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: datasets\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    dataset = load_dataset(dataset_name, split='train')
    data = dataset.to_pandas()

    if {TEXT_COLUMN, LABEL_COLUMN}.issubset(data.columns):
        return clean_dataset(data[[TEXT_COLUMN, LABEL_COLUMN]])

    data[LIX_COLUMN] = data.apply(extract_lix, axis=1)

    if label_scheme == 'lix-category':
        data[LABEL_COLUMN] = data.apply(extract_lix_category, axis=1)
    else:
        data[LABEL_COLUMN] = data[LIX_COLUMN].map(cefr_label_from_lix)

    return clean_dataset(data[[TEXT_COLUMN, LABEL_COLUMN, LIX_COLUMN]])


def extract_lix(row: pd.Series) -> float | None:
    '''Get LIX from either flattened or nested dataset columns.'''
    pd = load_pandas()

    if 'metrics_lix' in row and pd.notna(row['metrics_lix']):
        return float(row['metrics_lix'])

    metrics = row.get('metrics')
    if isinstance(metrics, dict) and metrics.get('lix') is not None:
        return float(metrics['lix'])

    return None


def extract_lix_category(row: pd.Series) -> str | None:
    '''Get the dataset's original LIX category when available.'''
    pd = load_pandas()

    if 'metrics_lix_category' in row and pd.notna(row['metrics_lix_category']):
        return pretty_label(row['metrics_lix_category'])

    metrics = row.get('metrics')
    if isinstance(metrics, dict) and metrics.get('lix_category') is not None:
        return pretty_label(metrics['lix_category'])

    return None


def cefr_label_from_lix(lix: float | None) -> str | None:
    '''Map LIX to approximate CEFR reading levels.'''
    pd = load_pandas()

    if lix is None or pd.isna(lix):
        return None
    if lix < 25:
        return 'A1'
    if lix < 30:
        return 'A2'
    if lix < 35:
        return 'B1'
    if lix < 45:
        return 'B2'
    if lix < 55:
        return 'C1'
    return 'C2'


def pretty_label(value: str) -> str:
    '''Convert labels like very_difficult to Very Difficult.'''
    return value.replace('_', ' ').title()


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    '''Remove empty text or label rows.'''
    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str).str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip()
    data = data[(data[TEXT_COLUMN] != '') & (data[LABEL_COLUMN] != '')]

    if data.empty:
        raise ValueError('Dataset has no usable rows after cleaning.')

    return data


def demo_dataset() -> pd.DataFrame:
    '''Small illustrative examples only, useful for checking that the code runs.'''
    pd = load_pandas()

    rows = [
        ('Katten sover. Solen skiner.', 'A1'),
        ('Jag går till skolan varje dag.', 'A1'),
        ('Barnet äter frukost och packar sin väska.', 'A2'),
        ('Vi träffar våra vänner i parken efter skolan.', 'A2'),
        ('Kommunen bygger en ny lekplats nära biblioteket.', 'B1'),
        ('Rapporten visar att fler elever behöver stöd i matematik.', 'B1'),
        ('Projektet ska utvärderas innan nästa etapp börjar.', 'B2'),
        ('Kursen passar personer som redan har grundläggande kunskaper.', 'B2'),
        ('Utredningen analyserar konsekvenserna av den föreslagna regeländringen.', 'C1'),
        ('Myndigheten efterfrågar en fördjupad konsekvensanalys innan beslut fattas.', 'C1'),
        ('Författningskommentaren preciserar hur begreppet väsentlig påverkan bör förstås i rättstillämpningen.', 'C2'),
        ('Åtgärdens proportionalitet måste prövas i relation till dokumenterade behov och alternativa handlingsvägar.', 'C2'),
    ]
    return pd.DataFrame(rows, columns=[TEXT_COLUMN, LABEL_COLUMN])


def load_embedding_model(model_name: str):
    '''Load a sentence-transformers model.'''
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: sentence-transformers\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    model = SentenceTransformer(model_name)
    try:
        transformer = model._first_module()
        input_names = transformer.tokenizer.model_input_names
        transformer.tokenizer.model_input_names = [
            name for name in input_names if name != 'token_type_ids'
        ]
    except AttributeError:
        pass

    return model


def create_embeddings(model, texts: list[str]):
    '''Encode texts with a sentence-transformers model.'''
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def likely_matches(probabilities, label_names: list[str], threshold: float, top_k: int) -> list[tuple[str, float]]:
    '''Return labels above the threshold, plus enough top labels to be useful.'''
    ranked = sorted(
        zip(label_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    matches = [(label, score) for label, score in ranked if score >= threshold]
    if len(matches) < top_k:
        matches = ranked[:top_k]
    return matches


def print_cefr_matches(
    texts: list[str],
    classifier,
    label_encoder: LabelEncoder,
    embedding_model,
    threshold: float,
    top_k: int,
) -> None:
    '''Print likely CEFR-level matches for each input text.'''
    label_names = label_encoder.classes_.tolist()
    embeddings = create_embeddings(embedding_model, texts)
    probabilities = classifier.predict_proba(embeddings)

    print('\nCEFR matches')
    for text, scores in zip(texts, probabilities):
        matches = likely_matches(scores, label_names, threshold, top_k)
        formatted = ', '.join(f'{label}: {score:.2f}' for label, score in matches)
        print(f'- text={text}')
        print(f'  matches={formatted}')


def classifier_candidates() -> dict[str, object]:
    '''Return several classifiers for comparison.'''
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: scikit-learn\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced'),
        ),
        'Linear SVM': make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
        ),
        'KNN': make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
    }


def train_classifiers(classifiers: dict[str, object], x_train, y_train, x_test, y_test) -> pd.DataFrame:
    '''Train classifiers and return evaluation scores.'''
    pd = load_pandas()

    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: scikit-learn\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    rows = []
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        rows.append(
            {
                'classifier': name,
                'accuracy': accuracy_score(y_test, predictions),
                'macro_f1': f1_score(y_test, predictions, average='macro'),
            }
        )
    return pd.DataFrame(rows).sort_values(['macro_f1', 'accuracy'], ascending=False)


def train_and_evaluate(
    data: pd.DataFrame,
    model_name: str,
    test_size: float,
    texts_to_classify: list[str],
    match_threshold: float,
    top_k: int,
    save_model_dir: Path | None,
) -> None:
    pd = load_pandas()

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: scikit-learn\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    label_counts = data[LABEL_COLUMN].value_counts()
    too_small = label_counts[label_counts < 2]
    if not too_small.empty:
        labels = ', '.join(too_small.index.tolist())
        raise ValueError(
            'Each label needs at least two examples for train/test splitting. '
            f'Add more data for: {labels}'
        )

    texts = data[TEXT_COLUMN].tolist()
    labels = data[LABEL_COLUMN].tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    print(f'Embedding model: {model_name}')
    embedding_model = load_embedding_model(model_name)
    x_train = create_embeddings(embedding_model, x_train_text)
    x_test = create_embeddings(embedding_model, x_test_text)

    classifiers = classifier_candidates()
    scores = train_classifiers(classifiers, x_train, y_train, x_test, y_test)
    best_name = scores.iloc[0]['classifier']
    classifier = classifiers[best_name]
    predictions = classifier.predict(x_test)

    label_names = label_encoder.classes_
    accuracy = accuracy_score(y_test, predictions)

    print('\nResults')
    print(f'Examples: {len(data)}')
    print('Labels: ' + ', '.join(label_names))
    print(f'Best classifier: {best_name}')
    print(f'Accuracy: {accuracy:.3f}\n')

    print('Classifier comparison')
    print(scores.to_string(index=False, formatters={'accuracy': '{:.3f}'.format, 'macro_f1': '{:.3f}'.format}))
    print()

    print('Classification report')
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(y_test, predictions), index=label_names, columns=label_names))

    if save_model_dir is not None:
        save_model_artifacts(
            save_model_dir,
            classifier,
            label_encoder,
            embedding_model,
            texts,
            y,
            model_name,
            best_name,
            scores,
            test_size,
        )

    print('\nExample test-set matches')
    print_cefr_matches(
        x_test_text[:5],
        classifier,
        label_encoder,
        embedding_model,
        match_threshold,
        top_k,
    )

    if texts_to_classify:
        print_cefr_matches(
            texts_to_classify,
            classifier,
            label_encoder,
            embedding_model,
            match_threshold,
            top_k,
        )


def train_and_evaluate_explicit_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_name: str,
    texts_to_classify: list[str],
    match_threshold: float,
    top_k: int,
    save_model_dir: Path | None,
) -> None:
    pd = load_pandas()

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: scikit-learn\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    train_labels = set(train_data[LABEL_COLUMN])
    test_labels = set(test_data[LABEL_COLUMN])
    if train_labels != test_labels:
        raise ValueError('Train and test datasets must contain the same label set.')

    train_data = train_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data[LABEL_COLUMN].tolist())
    y_test = label_encoder.transform(test_data[LABEL_COLUMN].tolist())
    x_train_text = train_data[TEXT_COLUMN].tolist()
    x_test_text = test_data[TEXT_COLUMN].tolist()

    print(f'Embedding model: {model_name}')
    embedding_model = load_embedding_model(model_name)
    x_train = create_embeddings(embedding_model, x_train_text)
    x_test = create_embeddings(embedding_model, x_test_text)

    classifiers = classifier_candidates()
    scores = train_classifiers(classifiers, x_train, y_train, x_test, y_test)
    best_name = scores.iloc[0]['classifier']
    classifier = classifiers[best_name]
    predictions = classifier.predict(x_test)

    label_names = label_encoder.classes_
    accuracy = accuracy_score(y_test, predictions)

    print('\nResults')
    print(f'Training examples: {len(train_data)}')
    print(f'Test examples: {len(test_data)}')
    print('Labels: ' + ', '.join(label_names))
    print(f'Best classifier: {best_name}')
    print(f'Accuracy: {accuracy:.3f}\n')

    print('Classifier comparison')
    print(scores.to_string(index=False, formatters={'accuracy': '{:.3f}'.format, 'macro_f1': '{:.3f}'.format}))
    print()

    print('Classification report')
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(y_test, predictions), index=label_names, columns=label_names))

    if save_model_dir is not None:
        save_model_artifacts(
            save_model_dir,
            classifier,
            label_encoder,
            embedding_model,
            x_train_text,
            y_train,
            model_name,
            best_name,
            scores,
            0.0,
        )

    if texts_to_classify:
        print_cefr_matches(
            texts_to_classify,
            classifier,
            label_encoder,
            embedding_model,
            match_threshold,
            top_k,
        )


def save_model_artifacts(
    output_dir: Path,
    classifier,
    label_encoder,
    embedding_model,
    texts: list[str],
    labels,
    model_name: str,
    best_classifier: str,
    scores: pd.DataFrame,
    test_size: float,
) -> None:
    '''Retrain the best classifier on all examples and save reusable artifacts.'''
    try:
        import joblib
    except ImportError as error:
        raise SystemExit(
            'Missing dependency: joblib\n'
            'Install project dependencies with: python3 -m pip install -r requirements.txt'
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nSaving trained model artifacts to: {output_dir}')
    full_embeddings = create_embeddings(embedding_model, texts)
    classifier.fit(full_embeddings, labels)

    joblib.dump(classifier, output_dir / 'classifier.joblib')
    joblib.dump(label_encoder, output_dir / 'label_encoder.joblib')
    scores.to_csv(output_dir / 'evaluation.csv', index=False)

    metadata = {
        'task': 'Swedish CEFR text classification',
        'embedding_model': model_name,
        'best_classifier': best_classifier,
        'labels': label_encoder.classes_.tolist(),
        'training_examples': len(texts),
        'test_size': test_size,
        'artifact_files': [
            'classifier.joblib',
            'label_encoder.joblib',
            'evaluation.csv',
            'metadata.json',
        ],
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a Swedish CEFR level classifier using sentence embeddings.'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=None,
        help='Path to CSV/TSV with columns: text,label',
    )
    parser.add_argument(
        '--train-data',
        type=Path,
        default=None,
        help='Path to explicit training CSV/TSV with columns: text,label',
    )
    parser.add_argument(
        '--test-data',
        type=Path,
        default=None,
        help='Path to explicit test CSV/TSV with columns: text,label',
    )
    parser.add_argument(
        '--huggingface',
        action='store_true',
        help='Load a dataset from Hugging Face',
    )
    parser.add_argument(
        '--hf-dataset',
        default=HF_DATASET,
        help='Hugging Face dataset repo id',
    )
    parser.add_argument(
        '--label-scheme',
        choices=['cefr', 'lix-category'],
        default='cefr',
        help='Use approximate CEFR labels or the original LIX categories',
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help='Sentence-transformers model name',
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.25,
        help='Fraction of data used for testing',
    )
    parser.add_argument(
        '--text',
        action='append',
        default=[],
        help='Text to classify after training. Can be used multiple times',
    )
    parser.add_argument(
        '--match-threshold',
        type=float,
        default=0.20,
        help='Show every CEFR label with probability at or above this value',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Minimum number of CEFR matches to show',
    )
    parser.add_argument(
        '--save-model-dir',
        type=Path,
        default=None,
        help='Directory where the best trained classifier and metadata are saved',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_data is not None or args.test_data is not None:
        if args.train_data is None or args.test_data is None:
            raise SystemExit('Use --train-data and --test-data together.')

        train_data = load_csv_dataset(args.train_data)
        test_data = load_csv_dataset(args.test_data)
        train_and_evaluate_explicit_split(
            train_data,
            test_data,
            model_name=args.model,
            texts_to_classify=args.text,
            match_threshold=args.match_threshold,
            top_k=args.top_k,
            save_model_dir=args.save_model_dir,
        )
        return

    if args.huggingface:
        data = load_huggingface_dataset(args.hf_dataset, args.label_scheme)
    else:
        data = load_csv_dataset(args.data)

    if args.data is None and not args.huggingface:
        print('No dataset supplied; using tiny illustrative demo data.')
        print('For the assignment, use your CEFR dataset or the Hugging Face text complexity dataset.\n')

    train_and_evaluate(
        data,
        model_name=args.model,
        test_size=args.test_size,
        texts_to_classify=args.text,
        match_threshold=args.match_threshold,
        top_k=args.top_k,
        save_model_dir=args.save_model_dir,
    )


if __name__ == '__main__':
    main()
