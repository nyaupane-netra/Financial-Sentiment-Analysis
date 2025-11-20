import os
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_PATH = "custom_financial_bert"

@st.cache_resource
def load_model_and_tokenizer():

    tokenizer = BertTokenizerFast.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.to("cpu")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()

id2label = {0: "negative", 1: "neutral", 2: "positive"}


def predict_with_probs(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1).numpy()[0]
    pred_idx = int(probs.argmax())

    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2]),
        "predicted_label": id2label[pred_idx]
    }


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("📈 Financial Sentiment Analysis (Custom BERT Model)")
st.write("Enter financial text below to predict sentiment:")

text = st.text_area("Text:", height=150)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        result = predict_with_probs(text)

        st.subheader("Prediction")
        st.write(f"### Sentiment: **{result['predicted_label'].upper()}**")

        st.subheader("Probabilities")
        st.write(result)

        st.bar_chart({
            "negative": result["negative"],
            "neutral": result["neutral"],
            "positive": result["positive"]
        })

st.caption("Custom BERT model trained on financial datasets.")
