import streamlit as st
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Vet-AI Pro", page_icon="🐾", layout="wide")

DATA_PATH = os.path.join('data', 'processed_data.json')

@st.cache_resource
def load_data_and_vectorize():
    if not os.path.exists(DATA_PATH):
        return None, None, None
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    # Create the TF-IDF Vectorizer
    # We use stop_words to ignore 'the', 'is', etc.
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Extract all symptoms from our dataset to "train" the vectorizer
    all_symptoms = [item['symptoms'] for item in data]
    tfidf_matrix = vectorizer.fit_transform(all_symptoms)
    
    return data, vectorizer, tfidf_matrix

data, vectorizer, tfidf_matrix = load_data_and_vectorize()

# --- 2. UI HEADER ---
st.title("🐾 Vet-AI: Advanced First-Aid System")
st.markdown("### Using TF-IDF Vector Space Modeling for Symptom Mapping")

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Describe the animal's symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if data is None:
        response = "Error: Please run prepare_data.py first."
    else:
        # TRANSFORM user input into the same TF-IDF space
        query_vec = vectorizer.transform([prompt])
        
        # CALCULATE Cosine Similarity between user query and ALL symptoms in dataset
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Find the best match
        best_match_idx = np.argmax(similarity)
        confidence = similarity[best_match_idx]

        if confidence > 0.20:  # Much more flexible threshold than 0.75
            match = data[best_match_idx]
            
            # --- CONSTRUCTING THE RESPONSE ---
            response = f"**Identified Condition:** {match['disease']}\n\n"
            response += f"### 🟢 Primary Intervention (First Aid)\n{match['primary']}\n\n"
            response += f"### 🏥 Secondary Intervention (Medical Therapy)\n{match['secondary']}\n"
            response += f"\n*(Confidence Score: {confidence:.2f})*"
        else:
            response = "I am not confident enough to give medical advice for this. Please contact a vet immediately."

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})