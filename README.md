<p align="center">
  <h1 align="center">Next Word Predictor</h1>
  <p align="center">
    <strong>Real-time word prediction • Top 5 suggestions • Click to build sentences</strong>
  </p>
</p>

<h1 align="center">
  <img src="https://img.shields.io/badge/Next_Word_Predictor-LSTM_AI-FFD700?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=1a1a1a"/>
</h1>

<p align="center">
  <strong>An intelligent LSTM-powered web app that predicts the next word in real-time — just like your phone keyboard, but smarter!</strong>
</p>

<p align="center">
  <a href="#live-demo">Live Demo</a> • 
  <a href="#features">Features</a> • 
  <a href="#how-it-works">How It Works</a> • 
  <a href="#model-details">Model Details</a> • 
  <a href="#run-locally">Run Locally</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/github/stars/yourusername/next-word-predictor?style=social" alt="Stars"/>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=yourusername.next-word-predictor" alt="Visitors"/>
</p>

---

### Live Demo
[Try it instantly!](https://next-word-predictor-msaqiipdsdagyyf2b8ab5e.streamlit.app/)

---

### Features
- Real-time next-word prediction using **Bidirectional LSTM**  
- Top 5 suggestions with confidence percentages  
- Click any word → instantly adds to your sentence  
- Stunning glassmorphism + gradient UI  
- Zero warnings, super clean deployment  
- Mobile-friendly & dark mode ready  
- Perfect for writing, learning, or fun!

---

### How It Works

1. You type → e.g., `"once upon a"`
2. Text → tokenized → padded
3. Fed into trained LSTM model
4. Model predicts probability for 20,000+ words
5. Top 5 shown with % confidence
6. Click any → sentence grows → repeat forever!

> It’s like having an AI writing assistant in your browser!

---

### Model Details

- **Architecture**: Embedding → Bidirectional LSTM → LSTM → Dense (softmax)
- **Vocabulary Size**: ~20,000+ words
- **Max Sequence Length**: 20
- **Training**: Adam + Categorical Crossentropy
- **Framework**: TensorFlow / Keras

---

### How It Works  

1. You type a few words → e.g., `"once upon a"`
2. The text is tokenized using the saved tokenizer
3. Converted into sequences → padded to fixed length
4. Fed into the trained **LSTM model**
5. Model outputs probability distribution over **~20,000+ words**
6. Top 5 most likely words are shown with % confidence
7. Click any → sentence grows → repeat endlessly!

> It’s like having an AI co-writer sitting next to you!

---

### Model Architecture  

```text
Embedding Layer (input_dim=vocab_size, output_dim=100)
→ Bidirectional LSTM (128 units)
→ Dropout (0.5)
→ LSTM (128 units)
→ Dense (vocab_size, softmax)
```

- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Trained for 50+ epochs  
- Vocabulary: 20,000+ most frequent words  
- Max Sequence Length: 20  

---

### Project Structure  
```
next-word-predictor/
├── app.py                  Main Streamlit app
├── my_model.h5             Trained LSTM model
├── tokenizer.pickle        Saved word tokenizer
├── assets/                 Screenshots & banner
├── requirements.txt
└── README.md
```

---

### Run Locally  

```bash
git clone https://github.com/yourusername/next-word-predictor.git
cd next-word-predictor

python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

---

### Built With  
- [TensorFlow / Keras](https://tensorflow.org) – Deep Learning  
- [Streamlit](https://streamlit.io) – Beautiful Web App  
- [NumPy](https://numpy.org) – Numerical Computing  
- Love & Late Nights  

---

### Author  
**Your Name**  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github)](https://github.com/Asad-Aziz-001)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/asad-aziz-140p)

<p align="center">
  <br>
  <strong>If you like this project — give it a ⭐ Star on GitHub!</strong>
  <br><br>
  <img src="https://media.giphy.com/media/W5e0J5h9JWZ6/giphy.gif" width="80"/>
</p>
```
