# Experiments

## Experiment 1: TF-IDF (1–2-grams) + Logistic Regression

- Date: 2025-12-27
- Code: `sentiment_baseline.py`
- Vectorizer:
  - TfidfVectorizer
  - max_features = 5000
  - ngram_range = (1, 2)
  - stop_words = "english"
- Model:
  - LogisticRegression
  - max_iter = 1000
- Test size: 20% (train_test_split, stratify by sentiment)

**Result:**

- Test accuracy: 0.8635
- Notes:
  - Both classes (0 = negative, 1 = positive) have similar precision/recall.
  - Confusion matrix shows some confusion between positive and negative, but overall performance is strong for a first model.


## Experiment 2: TF-IDF (1-gram only) + Logistic Regression

- Date: 2025-12-27
- Code: `sentiment_unigrams.py`
- Vectorizer:
  - TfidfVectorizer
  - max_features = 5000
  - ngram_range = (1, 1)
  - stop_words = "english"
- Model:
  - LogisticRegression
  - max_iter = 1000
- Test size: 20%

**Result:**

- Test accuracy: 0.8625
- Notes:
  - Performance is very similar to Experiment 1 (1–2-grams).
  - Adding bigrams did not significantly change accuracy on this dataset.

