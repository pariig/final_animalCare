import streamlit as st
import torch
import random
import json
import os
# Correctly importing from the src directory
from src.model import NeuralNet
from src.nltk_utils import bag_of_words, tokenize

# 1. SETUP AND MODEL LOADING
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths for your modular structure
INTENTS_PATH = os.path.join('data', 'intents.json')
MODEL_PATH = os.path.join('models', 'chat_model.pth')

# Load the intents data
with open(INTENTS_PATH, 'r') as f:
    intents = json.load(f)

# Load the trained model data
# weights_only=False is required for loading custom classes like NeuralNet
try:
    data = torch.load(MODEL_PATH, map_location=device, weights_only=False)
except FileNotFoundError:
    st.error("Model file not found! Please run 'python3 train.py' first.")
    st.stop()

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model and load the weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# 2. STREAMLIT UI
st.set_page_config(page_title="Street Animal First-Aid AI", page_icon="🐾")
st.title("🐾 Street Animal First-Aid AI")
st.markdown("Immediate first-aid advice for stray animals when a vet isn't available.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is the animal's condition?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # PROCESS THE INPUT (Inference)
    sentence = tokenize(prompt)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Response Logic
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
    else:
        response = "I'm not exactly sure about this condition. Please maintain basic hygiene, keep the animal calm, and contact a local animal NGO immediately."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})