import csv
import joblib
from typing import Literal

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates


# ----- Models for JSON API -----

class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=3, max_length=5000)


class PredictionResponse(BaseModel):
    label: Literal["positive", "negative"]
    probability: float = Field(..., ge=0.0, le=1.0)


# ----- Load model and vectorizer -----

vectorizer, clf = joblib.load("model.joblib")


app = FastAPI(title="Movie Review Sentiment API")


# Tell FastAPI where your templates folder is
templates = Jinja2Templates(directory="templates")


# ----- HTML form endpoints -----

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Just render the page with no result yet
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    # Transform and predict
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0].max()

    # Append prediction to CSV
    with open("predictions.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text, int(pred), float(proba)])

    # Data to show on the page
    result = {
        "label": int(pred),
        "confidence": float(proba),
        "input_text": text,
    }

    # Render the same page but now with result filled in
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result},
    )


# ----- JSON info + prediction API -----

@app.get("/info")
def info():
    return {
        "message": "Movie Review Sentiment API is running.",
        "html_form": "/",
        "predict_form": "/predict",
        "predict_json": "/api/predict",
        "docs": "/docs",
    }


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Predict sentiment from a movie review (JSON API)",
)
def predict_json(payload: ReviewRequest):
    text = payload.review_text.strip()

    if not text:
        # Valid JSON, but empty after stripping spaces
        raise HTTPException(status_code=400, detail="review_text must not be empty.")

    # Transform and predict with existing model
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    proba = float(clf.predict_proba(X)[0].max())

    # Map 0/1 to human-readable label
    label = "positive" if int(pred) == 1 else "negative"

    return PredictionResponse(label=label, probability=proba)
