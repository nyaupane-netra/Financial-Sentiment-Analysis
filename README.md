# Financial Sentiment Analysis

[![CI](https://github.com/nyaupane-netra/Financial-Sentiment-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/nyaupane-netra/Financial-Sentiment-Analysis/actions/workflows/ci.yml)

This project classifies financial text as **negative**, **neutral**, or
**positive**. It fine-tunes a custom BERT model and compares its predictions
with [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert).

## Features

- Custom BERT training notebook
- Streamlit dashboard for custom-model predictions
- Streamlit dashboard comparing the custom model with FinBERT
- FastAPI endpoint for programmatic predictions
- Correct label mapping for checkpoints with different class orders
- Automated tests and GitHub Actions checks

## How it works

```text
Financial text
      ↓
Streamlit dashboard or FastAPI
      ↓
Input validation and tokenization
      ↓
Custom BERT checkpoint
      ↓
Negative, neutral, or positive prediction
```

Application code is organized under `financial_sentiment/`:

- `core.py` handles validation, prediction, and label mapping.
- `service.py` loads and reuses the custom model.
- `config.py` reads the model path and inference device.
- `api.py` defines the HTTP endpoints.

## Project structure

```text
.
├── financial_sentiment/
│   ├── api.py
│   ├── config.py
│   ├── core.py
│   └── service.py
├── tests/
├── app.py
├── compare_app.py
├── custom_financial_bert_local.ipynb
├── Sentences_50Agree.txt
├── stock_data.csv
├── requirements.txt
├── requirements-training.txt
└── requirements-dev.txt
```

## Installation

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The applications expect a trained checkpoint in
`custom_financial_bert/`. You can use another location by setting:

```bash
export FIN_SENTIMENT_MODEL_PATH=/absolute/path/to/checkpoint
```

## Train the custom model

```bash
pip install -r requirements-training.txt
jupyter lab custom_financial_bert_local.ipynb
```

Run the notebook cells in order. The notebook combines the two included
datasets, creates a deterministic train/test split, fine-tunes
`bert-base-uncased`, calculates accuracy and weighted F1, and saves the model to
`custom_financial_bert/`.

## Run the applications

Custom-model dashboard:

```bash
streamlit run app.py
```

Custom BERT versus FinBERT dashboard:

```bash
streamlit run compare_app.py
```

FastAPI service:

```bash
uvicorn financial_sentiment.api:app --reload
```

Then open `http://localhost:8000/docs` to try the API interactively.

Example request:

```bash
curl -X POST http://localhost:8000/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{"text":"The company raised its full-year revenue forecast."}'
```

## Labels and data

The custom model uses:

| Model ID | Sentiment |
| ---: | --- |
| 0 | negative |
| 1 | neutral |
| 2 | positive |

`stock_data.csv` contains `Text` and `Sentiment` columns with source labels
`-1`, `0`, and `1`. `Sentences_50Agree.txt` stores one sentence per line with
an `@negative`, `@neutral`, or `@positive` suffix.

The original dataset sources and licenses should be confirmed before commercial
use or redistribution.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m pytest -q
```

The tests cover label mapping, input validation, missing checkpoints, model
readiness, and the API response contract. They do not download model weights.

## Contributors

Developed collaboratively by
[`nyaupane-netra`](https://github.com/nyaupane-netra) and
[`neupaneb`](https://github.com/neupaneb).

## Limitations

- The final evaluation results and confusion matrix are not yet documented.
- Predictions may fail on sarcasm, unfamiliar terminology, or text requiring
  broader context.
- Confidence scores have not been calibrated.
- This project is educational and is not financial advice.
