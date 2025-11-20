import os
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)

# ======================================================
# MUST BE FIRST STREAMLIT COMMAND
# ======================================================
st.set_page_config(page_title="FinSentiment Comparison", layout="wide")

# ---------- Environment fixes ----------
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = torch.device("cpu")

CUSTOM_MODEL_PATH = "custom_financial_bert"

# -----------------------------------------------------
# Load models
# -----------------------------------------------------
@st.cache_resource
def load_models():
    # Custom model
    custom_tok = AutoTokenizer.from_pretrained(
        CUSTOM_MODEL_PATH,
        local_files_only=True
    )
    custom_mod = AutoModelForSequenceClassification.from_pretrained(
        CUSTOM_MODEL_PATH,
        local_files_only=True
    ).to(DEVICE)
    custom_mod.eval()

    # FinBERT
    fin_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fin_mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)
    fin_mod.eval()

    return custom_tok, custom_mod, fin_tok, fin_mod


custom_tokenizer, custom_model, finbert_tokenizer, finbert_model = load_models()
id2label = {0: "negative", 1: "neutral", 2: "positive"}


# -----------------------------------------------------
# Prediction function
# -----------------------------------------------------
def predict(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    return id2label[pred_idx], probs


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("📊 Financial Sentiment Comparison Dashboard")
st.write("Compare your **Custom BERT Model** with **ProsusAI/FinBERT** on any text.")

text = st.text_area("Enter financial or general text:", height=140)

if st.button("Compare Models"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        custom_label, custom_probs = predict(text, custom_tokenizer, custom_model)
        fin_label, fin_probs = predict(text, finbert_tokenizer, finbert_model)

        col1, col2 = st.columns(2)

        # ----------------- Custom Model -----------------
        with col1:
            st.header("🔵 Your Custom BERT Model")
            st.subheader(f"Prediction: **{custom_label.upper()}**")

            st.bar_chart({
                "negative": custom_probs[0],
                "neutral": custom_probs[1],
                "positive": custom_probs[2],
            })

            st.write("### Probability Values")
            st.write(f"- Negative: **{custom_probs[0]:.4f}**")
            st.write(f"- Neutral: **{custom_probs[1]:.4f}**")
            st.write(f"- Positive: **{custom_probs[2]:.4f}**")

        # ----------------- FinBERT -----------------
        with col2:
            st.header("🟢 ProsusAI / FinBERT")
            st.subheader(f"Prediction: **{fin_label.upper()}**")

            st.bar_chart({
                "negative": fin_probs[0],
                "neutral": fin_probs[1],
                "positive": fin_probs[2],
            })

            st.write("### Probability Values")
            st.write(f"- Negative: **{fin_probs[0]:.4f}**")
            st.write(f"- Neutral: **{fin_probs[1]:.4f}**")
            st.write(f"- Positive: **{fin_probs[2]:.4f}**")

        st.write("---")
        st.subheader("📌 Summary")

        if custom_label == fin_label:
            st.success(f"Both models agree: **{custom_label}**")
        else:
            st.error(
                f"Models disagree!\n\n"
                f"Your model → **{custom_label}**\n"
                f"FinBERT → **{fin_label}**"
            )

st.caption("NLP Project • Custom Financial Model vs FinBERT Comparison")
