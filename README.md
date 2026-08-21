# Financial Sentiment Analysis

An end-to-end NLP project that fine-tunes BERT to classify financial text as
**negative**, **neutral**, or **positive**, then compares the custom checkpoint
with [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) in an
interactive Streamlit dashboard.

## What this project demonstrates

- Combining and normalizing two labeled financial-text datasets
- Fine-tuning `bert-base-uncased` for three-class sequence classification
- Serving a local Hugging Face checkpoint with Streamlit
- Comparing two models without assuming they share the same label order
- Testing inference utilities and running lint/tests in GitHub Actions

## Repository structure

```text
.
├── .github/workflows/ci.yml        # automated linting and tests
├── app.py                          # custom-model Streamlit dashboard
├── compare_app.py                  # custom BERT vs FinBERT dashboard
├── inference.py                    # shared, tested inference logic
├── custom_financial_bert_local.ipynb
├── Sentences_50Agree.txt           # Financial PhraseBank samples
├── stock_data.csv                  # labeled stock-related text
├── requirements.txt                # application dependencies
├── requirements-training.txt       # notebook/training dependencies
└── tests/test_inference.py
```

Model weights and training outputs are intentionally excluded from Git because
they are large and reproducible from the notebook.

## Label mapping

The custom model uses:

| Model ID | Sentiment |
| ---: | --- |
| 0 | negative |
| 1 | neutral |
| 2 | positive |

`stock_data.csv` starts with labels `-1`, `0`, and `1`; the notebook remaps
them to `0`, `1`, and `2`. The comparison app reads labels from each model's
configuration, which prevents incorrect FinBERT probability labels.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The apps require a trained checkpoint at `custom_financial_bert/`. Generate it
by running the notebook as described below.

### Run the custom-model dashboard

```bash
streamlit run app.py
```

### Compare the custom model with FinBERT

```bash
streamlit run compare_app.py
```

The comparison dashboard downloads `ProsusAI/finbert` from Hugging Face on its
first run. Both dashboards currently use CPU inference for portability.

## Train the custom model

Install the training environment:

```bash
pip install -r requirements-training.txt
jupyter lab custom_financial_bert_local.ipynb
```

Run the notebook from top to bottom. It:

1. Reads `stock_data.csv` and `Sentences_50Agree.txt`.
2. Normalizes their labels and combines the datasets.
3. Creates a deterministic 80/20 train/test split (`random_state=42`).
4. Tokenizes text with `bert-base-uncased`.
5. Fine-tunes for three epochs and reports accuracy and weighted F1.
6. Saves the checkpoint and tokenizer to `custom_financial_bert/`.

> The repository does not currently publish a final benchmark table. Re-run the
> notebook in a fixed environment before presenting performance claims, and add
> the resulting test metrics and confusion matrix here.

## Data

`stock_data.csv` expects `Text` and `Sentiment` columns. The phrase-bank file
uses one sentence per line with an `@negative`, `@neutral`, or `@positive`
suffix.

Before redistributing or using these datasets commercially, verify the original
sources and their licenses. Dataset provenance and licensing should be added to
this README when the original download links are confirmed.

## Development and tests

The lightweight test suite does not download model weights.

```bash
pip install -r requirements-dev.txt
ruff check app.py compare_app.py inference.py tests
python -m pytest -q
```

GitHub Actions runs these checks on every push and pull request.

## Collaboration

This project was developed collaboratively by
[`nyaupane-netra`](https://github.com/nyaupane-netra) and
[`neupaneb`](https://github.com/neupaneb). Git history is retained to show both
contributors' work. Add component-level ownership here only when both
contributors have confirmed the division of responsibilities.

## Limitations

- Predictions reflect patterns in the training data and may fail on new market
  language, sarcasm, or text requiring broader context.
- Confidence scores are not guaranteed to be calibrated probabilities.
- The model is an educational classifier, not financial advice or a trading
  signal.
- The current notebook uses a single train/test split; stronger evaluation
  should include a validation set, class-level metrics, and error analysis.

## Troubleshooting

- **Local model not found:** run the training notebook and confirm that
  `custom_financial_bert/config.json` exists.
- **FinBERT cannot download:** confirm internet access, then retry the comparison
  app.
- **Training is slow:** reduce batch size or sequence length, or use a supported
  GPU runtime.
