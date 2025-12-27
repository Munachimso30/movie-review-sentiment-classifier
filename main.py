import csv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os

vectorizer, clf = joblib.load("model.joblib")

app = FastAPI(title="Movie Review Sentiment API")

class Review(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(review: Review):
    X = vectorizer.transform([review.text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0].max()

    # Append to CSV file
    with open("predictions.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([review.text, int(pred), float(proba)])

    return {
        "label": int(pred),
        "confidence": float(proba),
        "input_text": review.text,
    }

