# app.py - FINAL VERSION WITH DEVELOPER CARD
import os
import warnings
import logging

# Kill ALL warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# ============================
# Page Config & Styling
# ============================
st.set_page_config(page_title="Next Word Predictor", page_icon="crystal_ball", layout="centered")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .title {font-size: 3.5rem !important; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #FFD700, #FFA500);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: #e0e0e0; font-size: 1.3rem; margin-bottom: 2rem;}
    .prediction-box {background: rgba(255,255,255,0.15); backdrop-filter: blur(12px);
                     padding: 1.8rem; border-radius: 18px; text-align: center;
                     border: 1px solid rgba(255,255,255,0.25); margin: 2rem 0;}
    .chip {display: inline-block; background: #4CAF50; color: white;
           padding: 0.7rem 1.4rem; margin: 0.5rem; border-radius: 50px;
           font-weight: bold; cursor: pointer; transition: 0.3s;}
    .chip:hover {background: #45a049; transform: translateY(-3px);}
</style>
""", unsafe_allow_html=True)

# ============================
# Session State
# ============================
if "text" not in st.session_state:
    st.session_state.text = "the quick brown"

# ============================
# Load Model
# ============================
@st.cache_resource(show_spinner="Loading the AI brain...")
def load_artifacts():
    model = load_model("my_model.h5", compile=False)
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ============================
# Prediction Function
# ============================
def predict_top_k(input_text, top_k=5, max_sequence_len=20):
    if not input_text.strip():
        return []
    seq = tokenizer.texts_to_sequences([input_text.lower()])[0]
    seq = pad_sequences([seq], maxlen=max_sequence_len-1, padding='pre')
    probs = model.predict(seq, verbose=0)[0]
    top_idx = np.argsort(probs)[-top_k:][::-1]
    results = []
    for i in top_idx:
        word = next((w for w, idx in tokenizer.word_index.items() if idx == i), None)
        if word and probs[i] > 0.01:
            results.append((word, round(probs[i]*100, 2)))
    return results

MAX_SEQUENCE_LEN = 20

# ============================
# Main UI
# ============================
st.markdown("<h1 class='title'>Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Type anything and let AI complete your thoughts!</p>", unsafe_allow_html=True)

user_input = st.text_input(
    "Start typing here...",
    value=st.session_state.text,
    placeholder="e.g. once upon a time",
    key="input_box"
)

if user_input != st.session_state.text:
    st.session_state.text = user_input

if st.button("Predict Next Words", type="primary", use_container_width=True) or st.session_state.text.strip():
    predictions = predict_top_k(st.session_state.text, top_k=5, max_sequence_len=MAX_SEQUENCE_LEN)

    if predictions:
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown("### Click any word to continue:")
        cols = st.columns(len(predictions))
        for i, (word, prob) in enumerate(predictions):
            with cols[i]:
                if st.button(f"**{word}**\n{prob}%", key=f"suggest_{i}", use_container_width=True):
                    st.session_state.text += " " + word
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        best = predictions[0][0]
        st.success(f"**Best next word:** `{best}` → **{st.session_state.text} {best}**")
    else:
        st.info("No strong prediction yet. Try adding more words!")

# ============================
# Sidebar - Model Info + Developer Card
# ============================
with st.sidebar:
    st.markdown("### Model Information")
    st.success("Model loaded successfully!")
    st.write(f"**Vocabulary Size:** {len(tokenizer.word_index):,}")
    st.write(f"**Max Sequence Length:** {MAX_SEQUENCE_LEN}")
    st.write(f"**Architecture:** Bidirectional LSTM")
    st.caption("Zero warnings • Super fast • Professional deployment")

    st.markdown("---")

    # === YOUR DEVELOPER CARD IN SIDEBAR ===
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 1.6rem;
        border-radius: 18px;
        border: 1.5px solid rgba(255,255,255,0.4);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        margin: 20px 0;
        text-align: center;
    ">
        <h3 style="color:white; margin:0 0 12px 0; font-size:1.6rem;">Developer</h3>
        <p style="color:#f0f0f0; font-size:1.1rem; line-height:1.6; margin:8px 0;">
            <strong style="font-size:1.35rem; color:#FFD700;">ASAD AZIZ</strong><br>
            BS Artificial Intelligence Student<br>
            AI Enthusiast • NLP • Deep Learning
        </p>
        <div style="margin-top:15px;">
            <a href="https://github.com/Asad-Aziz-001" target="_blank">
                <button style="
                    background:#333; color:white; padding:9px 18px; border-radius:12px; 
                    border:none; cursor:pointer; margin:5px; font-weight:bold; font-size:0.95rem;
                    box-shadow:0 4px 12px rgba(0,0,0,0.4); transition:0.3s;
                ">GitHub</button>
            </a>
            <a href="https://www.linkedin.com/in/asad-aziz-140p" target="_blank">
                <button style="
                    background:#0A66C2; color:white; padding:9px 18px; border-radius:12px; 
                    border:none; cursor:pointer; margin:5px; font-weight:bold; font-size:0.95rem;
                    box-shadow:0 4px 12px rgba(0,0,0,0.4); transition:0.3s;
                ">LinkedIn</button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown("<br><p style='text-align:center; color:#aaa; font-size:0.9rem;'>Made with ❤️ using Streamlit • LSTM • TensorFlow</p>", unsafe_allow_html=True)
