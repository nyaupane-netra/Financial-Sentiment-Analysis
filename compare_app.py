"""Compare the custom classifier with ProsusAI/FinBERT."""

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from financial_sentiment.core import predict, require_local_model

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent
CUSTOM_MODEL_PATH = ROOT / "custom_financial_bert"
FINBERT_ID = "ProsusAI/finbert"
DEVICE = torch.device("cpu")


@st.cache_resource
def load_models():
    custom_path = require_local_model(CUSTOM_MODEL_PATH)
    custom_tokenizer = AutoTokenizer.from_pretrained(custom_path, local_files_only=True)
    custom_model = AutoModelForSequenceClassification.from_pretrained(
        custom_path, local_files_only=True
    ).to(DEVICE)
    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_ID)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_ID).to(
        DEVICE
    )
    custom_model.eval()
    finbert_model.eval()
    return custom_tokenizer, custom_model, finbert_tokenizer, finbert_model


def render_result(title, result):
    st.header(title)
    st.subheader(f"Prediction: {result['predicted_label'].title()}")
    st.bar_chart(result["probabilities"])
    for label, probability in result["probabilities"].items():
        st.write(f"{label.title()}: **{probability:.2%}**")


st.title("📊 Financial Sentiment Model Comparison")
st.write("Compare the custom BERT classifier with ProsusAI/FinBERT.")
text = st.text_area(
    "Financial text",
    placeholder="Example: Quarterly earnings fell short of analyst expectations.",
    height=140,
)

if st.button("Compare models", type="primary"):
    if not text.strip():
        st.warning("Enter some financial text first.")
    else:
        try:
            custom_tokenizer, custom_model, finbert_tokenizer, finbert_model = (
                load_models()
            )
            custom_result = predict(text, custom_tokenizer, custom_model, DEVICE)
            finbert_result = predict(text, finbert_tokenizer, finbert_model, DEVICE)
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(str(error))
        else:
            left, right = st.columns(2)
            with left:
                render_result("Custom BERT", custom_result)
            with right:
                render_result("ProsusAI/FinBERT", finbert_result)

            if custom_result["predicted_label"] == finbert_result["predicted_label"]:
                st.success(
                    f"Both models agree: {custom_result['predicted_label'].title()}"
                )
            else:
                st.warning(
                    "The models disagree: "
                    f"Custom BERT predicts {custom_result['predicted_label']}; "
                    f"FinBERT predicts {finbert_result['predicted_label']}."
                )

st.caption("The first FinBERT run downloads its model files from Hugging Face.")
