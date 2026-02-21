from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("/sentiment_analysis/sentiment_model.pkl")
vectorizer = joblib.load("/sentiment_analysis/vectorizer.pkl")

@app.post("/analyze")
def analyze(text: str):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return {"sentiment": prediction}