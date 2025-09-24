import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# -----------------------------
# Load pretrained Hugging Face model
# -----------------------------
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# -----------------------------
# Helper function
# -----------------------------
def translate(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    # Generate translation
    translated_tokens = model.generate(**inputs)
    # Decode output tokens to string
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("English → French Translator (Hugging Face)")

english_text = st.text_input("Enter English sentence:")

if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter an English sentence.")
    else:
        french_translation = translate(english_text)
        st.success(f"French Translation: {french_translation}")
