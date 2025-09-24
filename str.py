import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# -----------------------------
# Load trained model and token data
# -----------------------------
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
input_token_index = data['input_token_index']
target_token_index = data['target_token_index']
reverse_target_char_index = data['reverse_target_char_index']
max_encoder_seq_length = data['max_encoder_seq_length']
max_decoder_seq_length = data['max_decoder_seq_length']
latent_dim = data['latent_dim']

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)

# -----------------------------
# Build encoder inference model
# -----------------------------
encoder_inputs = model.input[0]
encoder_lstm = model.layers[2]
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# -----------------------------
# Build decoder inference model
# -----------------------------
decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

decoder_inputs_inf = Input(shape=(None, num_decoder_tokens), name='decoder_input_inf')
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_h_inf')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_c_inf')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs_inf, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [state_h_dec, state_c_dec]

decoder_model = Model(
    [decoder_inputs_inf] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# -----------------------------
# Helper functions
# -----------------------------
def encode_input_text(input_text):
    encoder_input = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input[0, t, input_token_index[char]] = 1.
    for t in range(len(input_text), max_encoder_seq_length):
        encoder_input[0, t, input_token_index[' ']] = 1.
    return encoder_input

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.  # <start> token

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            break
        decoded_sentence += sampled_char
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("English → French Translator (Character-level LSTM)")

english_text = st.text_input("Enter English sentence:")

if st.button("Translate"):
    if english_text.strip() == "" or not all(c in input_token_index for c in english_text):
        st.warning("Please enter an English sentence.")
    else:
        input_seq = encode_input_text(english_text)
        french_translation = decode_sequence(input_seq)
        st.success(f"French Translation: {french_translation}")
