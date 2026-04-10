import streamlit as st
import numpy as np
from joblib import load
from feature_pipeline import query_point_creator_tfidf

# Load model & vectorizer
@st.cache_resource
def load_model():
    model = load("model_withtf.joblib")
    tfidf = load("tfidf_vectorizer.joblib")
    STOP_WORDS = load('stopwords.joblib')
    return model, tfidf, STOP_WORDS

model, tfidf,STOP_WORDS = load_model()

st.title("Duplicate Question Detector")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check"):
    if q1 and q2:
        features = query_point_creator_tfidf(q1, q2, tfidf,STOP_WORDS)
        result = model.predict(features)[0]

        
        if result == 1:
            st.success("✅ Duplicate Questions")
        else:
            st.error("❌ Not Duplicate Questions")
    else:
        st.warning("Please enter both questions")