import csv
import joblib
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load model and vectorizer
vectorizer, clf = joblib.load("model.joblib")

app = FastAPI(title="Movie Review Sentiment API")

# Tell FastAPI where your templates folder is
templates = Jinja2Templates(directory="templates")


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
