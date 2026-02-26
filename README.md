# Financial Sentiment Analysis

This project trains and serves a custom BERT model for **financial sentiment classification** (`negative`, `neutral`, `positive`) and compares it against **ProsusAI/FinBERT**.

## Project Contents

- `custom_financial_bert_local.ipynb` - end-to-end training and inference workflow
- `stock_data.csv` - labeled social-media-style stock sentiment data
- `Sentences_50Agree.txt` - Financial PhraseBank style labeled sentences
- `app.py` - Streamlit app for custom model inference
- `compare_app.py` - Streamlit app to compare custom model vs FinBERT

## Label Mapping

The project uses 3 sentiment labels:

- `0` -> `negative`
- `1` -> `neutral`
- `2` -> `positive`

In the notebook, `stock_data.csv` sentiment values are remapped as:

- `-1 -> 0` (`negative`)
- `0 -> 1` (`neutral`)
- `1 -> 2` (`positive`)

## Data Format

### `stock_data.csv`
Expected columns:

- `Text`
- `Sentiment` (values in `{-1, 0, 1}`)

### `Sentences_50Agree.txt`
One sample per line with label suffix:

- `... @negative`
- `... @neutral`
- `... @positive`

## Environment Setup

Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets evaluate scikit-learn pandas numpy streamlit
```

## Train the Custom Model

1. Open and run `custom_financial_bert_local.ipynb` cells in order.
2. The notebook trains `bert-base-uncased` with 3 output labels.
3. After training, it saves model artifacts to:

- `custom_financial_bert/`

The Streamlit apps expect this folder to exist locally.

## Run the Apps

### 1) Custom Model App

```bash
streamlit run app.py
```

- Loads local model from `custom_financial_bert/`
- Runs sentiment prediction with probability outputs

### 2) Comparison App (Custom vs FinBERT)

```bash
streamlit run compare_app.py
```

- Loads local custom model from `custom_financial_bert/`
- Downloads/loads `ProsusAI/finbert` from Hugging Face
- Shows side-by-side predictions and confidence bars

## Notes

- `custom_financial_bert/` is excluded in `.gitignore`, so model weights are not committed.
- If `compare_app.py` is run for the first time, internet access is needed to download `ProsusAI/finbert`.
- Both apps are set to CPU execution.

## Troubleshooting

- **Error: local model path not found**
  - Ensure `custom_financial_bert/` exists in the project root.
- **Import errors**
  - Reinstall dependencies in the active virtual environment.
- **Slow inference/training**
  - CPU-only mode can be slower; reduce batch sizes or text length if needed.
