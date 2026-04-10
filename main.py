import streamlit as st
import numpy as np
from joblib import load
from feature_pipeline import query_point_creator_tfidf
import requests

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
        url = "https://duplicate-question-pairs-nlp-rdmv.onrender.com/predict"

        try:
            with st.spinner("Checking..."):
                response = requests.post(url, json={
                    "q1": q1,
                    "q2": q2
                })

            result = response.json()

            if result["duplicate"] == 1:
                st.success("✅ Duplicate Questions")
            else:
                st.error("❌ Not Duplicate Questions")

        except Exception as e:
            st.error("API Error: " + str(e))

    else:
        st.warning("Please enter both questions")