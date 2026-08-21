"""Streamlit dashboard for the custom financial sentiment model."""

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from financial_sentiment.core import predict, require_local_model

st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="📈")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "custom_financial_bert"
DEVICE = torch.device("cpu")


@st.cache_resource
def load_model():
    model_path = require_local_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


st.title("📈 Financial Sentiment Analysis")
st.write("Classify financial text as negative, neutral, or positive.")
text = st.text_area(
    "Financial text",
    placeholder="Example: The company raised its full-year revenue forecast.",
    height=150,
)

if st.button("Analyze sentiment", type="primary"):
    if not text.strip():
        st.warning("Enter some financial text first.")
    else:
        try:
            tokenizer, model = load_model()
            result = predict(text, tokenizer, model, DEVICE)
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(str(error))
        else:
            st.subheader(f"Prediction: {result['predicted_label'].title()}")
            st.bar_chart(result["probabilities"])
            st.dataframe(
                {
                    "Sentiment": list(result["probabilities"]),
                    "Probability": list(result["probabilities"].values()),
                },
                hide_index=True,
                use_container_width=True,
            )

st.caption("Custom BERT model trained on two financial-sentiment datasets.")
