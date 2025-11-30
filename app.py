# app.py - FINAL VERSION (No warnings + Click-to-append works perfectly)
import os
import warnings
import logging

# Kill ALL TensorFlow/Keras/absl warnings
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
# Initialize session state
# ============================
if "text" not in st.session_state:
    st.session_state.text = "the quick brown"

# ============================
# Load Model (silently)
# ============================
@st.cache_resource(show_spinner="Loading the AI model...")
def load_artifacts():
    model = load_model("my_model.h5", compile=False)  # compile=False → no absl warning
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

MAX_SEQUENCE_LEN = 20  # Change if your model used different length

# ============================
# UI
# ============================
st.markdown("<h1 class='title'>Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Type and click any suggestion to continue!</p>", unsafe_allow_html=True)

# Text input (uses st.session_state.text)
user_input = st.text_input(
    "Start typing...",
    value=st.session_state.text,
    placeholder="e.g. once upon a",
    key="input_box"
)

# Update session state when user types
if user_input != st.session_state.text:
    st.session_state.text = user_input

# Predict button (or auto-predict on input)
if st.button("Predict Next Words", type="primary", use_container_width=True) or st.session_state.text.strip():
    predictions = predict_top_k(st.session_state.text, top_k=5, max_sequence_len=MAX_SEQUENCE_LEN)

    if predictions:
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown("### Click a word to add it:")
        
        cols = st.columns(len(predictions))
        for i, (word, prob) in enumerate(predictions):
            with cols[i]:
                if st.button(f"**{word}**\n{prob}%", key=f"suggest_{i}", use_container_width=True):
                    st.session_state.text += " " + word
                    st.rerun()  # Refresh to show updated text
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        best = predictions[0][0]
        st.success(f"**Best next word:** `{best}` → **{st.session_state.text} {best}**")
    else:
        st.info("No strong prediction. Try more words!")

# ============================
# Sidebar Info
# ============================
with st.sidebar:
    st.header("Model Info")
    st.success("Model loaded successfully!")
    st.write(f"**Tokenizer Vocabulary:** {len(tokenizer.word_index):,}")
    st.write(f"**Max Sequence Length:** {MAX_SEQUENCE_LEN}")
    
    st.divider()
    st.caption("Built with ❤️ using:")
    st.write("- TensorFlow / Keras")
    st.write("- Streamlit")
    st.write("- LSTM Neural Network")
    
st.markdown("---")
st.caption("Next Word Predictor • LSTM • Built with Streamlit + TensorFlow")