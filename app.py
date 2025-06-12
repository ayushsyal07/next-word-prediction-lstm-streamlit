import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM model
model = load_model('next_word_lstm.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "No prediction"

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Next Word Predictor", page_icon="🤖", layout="centered")

st.markdown("<h1 style='text-align: center;'>🔮 Next Word Prediction using LSTM</h1>", unsafe_allow_html=True)
st.markdown("Enter a sequence of words, and the model will predict the next word using an LSTM-based neural network.")

# Input field
input_text = st.text_input("📝 Enter a sequence", placeholder="e.g., To be or not to")

# Predict button
if st.button("🚀 Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter a valid sequence of words.")
    else:
        with st.spinner("Predicting next word..."):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success("Prediction complete!")
        st.markdown(f"### 🧠 Predicted Next Word: **`{next_word}`**")

# Optional: Footer or explanation
with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses an LSTM (Long Short-Term Memory) model trained on a text corpus to predict the next word 
    in a given sequence. It uses a tokenizer to convert words into tokens and generates the next most probable word.
    """)

