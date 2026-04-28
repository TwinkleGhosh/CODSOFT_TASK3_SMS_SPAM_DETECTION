import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from train import load_models
from predict import predict

# Page config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
)


# Load models
@st.cache_resource
def get_models():
    return load_models()


try:
    vectorizer, nb_model, lr_model = get_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# UI
st.title("🛡️ SMS Spam Detector")
st.caption("Ensemble model: Naive Bayes + Logistic Regression with TF-IDF bigrams")

if not models_loaded:
    st.error(
        "⚠️ Trained models not found. "
        "Please run `python src/train.py` first to generate the model files in `models/`."
    )
    st.stop()

# Input
sms_input = st.text_area(
    "Paste or type an SMS message:",
    placeholder="e.g. Congratulations! You've won a free prize. Call now!",
    height=120,
)

col1, col2 = st.columns([1, 5])
with col1:
    predict_btn = st.button("Analyse", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# Prediction
if predict_btn:
    if not sms_input.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict(sms_input, vectorizer, nb_model, lr_model)
        prob = result["spam_probability"]

        st.divider()

        if result["is_spam"]:
            st.error(f"### 🚫 SPAM detected  ({prob:.0%} confidence)")
        else:
            st.success(f"### ✅ Not Spam  ({(1 - prob):.0%} confidence it's safe)")

        # Probability bar
        st.write("**Spam probability**")
        st.progress(prob)

        # Detail expander
        with st.expander("Model details"):
            st.write(
                f"- Naive Bayes prediction : {'Spam' if result['nb_pred'] else 'Ham'}"
            )
            st.write(
                f"- Logistic Regression    : {'Spam' if result['lr_pred'] else 'Ham'}"
            )
            st.write(f"- Ensemble decision rule : spam if NB + LR ≥ 1")

#  Adversarial test section
st.divider()
with st.expander("🧪 Run adversarial test cases"):
    if st.button("Run tests"):
        from predict import ADVERSARIAL_CASES

        for msg in ADVERSARIAL_CASES:
            r = predict(msg, vectorizer, nb_model, lr_model)
            label = "🚫 Spam" if r["is_spam"] else "✅ Ham"
            st.write(f"**{msg}** → {label} `(p={r['spam_probability']:.2f})`")
