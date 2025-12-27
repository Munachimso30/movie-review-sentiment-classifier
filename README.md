# Movie Review Sentiment Classifier

Sentiment analysis on movie reviews using TF-IDF features and a logistic regression classifier, served via a FastAPI web app. The project exposes both an HTML form and a JSON API, and is deployed on Render for easy access. [file:1][web:136][web:142]

## Live demo

- Render URL: https://dashboard.render.com/web/srv-d5823r6r433s73f19veg
- GitHub repo: https://github.com/Munachimso30/movie-review-sentiment-classifier

## Project overview

- Binary sentiment classification (positive vs negative) on movie reviews.
- Text preprocessing and TF-IDF vectorization.
- Logistic regression model trained and saved with `joblib`.
- FastAPI app with:
  - HTML form at `/` and `/predict`.
  - JSON API at `/api/predict` with Pydantic validation.
- Predictions optionally logged to `predictions.csv`. [file:1][web:136][web:142]

## API endpoints

- `GET /`  
  - Renders the HTML form where you can paste a movie review and see the prediction.

- `POST /predict` (HTML form)  
  - Accepts form field `text` and renders the same HTML template with prediction results.

- `GET /info`  
  - Returns basic JSON service information and links to key endpoints.

- `POST /api/predict` (JSON)  
  - Request body (JSON):
    ```
    {
      "review_text": "This movie was amazing, I really enjoyed it!"
    }
    ```
  - Response body:
    ```
    {
      "label": "positive",
      "probability": 0.93
    }
    ```
  - Validation:
    - `review_text` must be a non-empty string between 3 and 5000 characters.
    - Returns 400 if the text is empty after trimming.
    - Returns 422 if the JSON is missing `review_text` or has the wrong type. [file:1][web:58][web:67][web:73]

## Running locally


See `experiments.md` for more details.

