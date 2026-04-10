from fastapi import FastAPI
from joblib import load
from feature_pipeline import query_point_creator_tfidf
import nltk

app = FastAPI()
nltk.download('stopwords')
# ✅ Load model + tfidf
model = load("model_withtf.joblib")
tfidf = load("tfidf_vectorizer.joblib")

# ✅ Load stopwords
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    q1 = data["q1"]
    q2 = data["q2"]

    features = query_point_creator_tfidf(q1, q2, tfidf, STOP_WORDS)

    pred = model.predict(features)[0]

    return {"duplicate": int(pred)}