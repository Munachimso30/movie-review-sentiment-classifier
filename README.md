# Movie Review Sentiment Classifier

## Overview

This project trains a machine learning model to classify movie reviews as positive (1) or negative (0). The model uses TF-IDF text features and a logistic regression classifier implemented in Python with scikit-learn. [web:46][web:53]

## Dataset

- Source: Binary movie review sentiment dataset (CSV) with two columns:
  - `review`: text of the movie review
  - `sentiment`: 0 = negative, 1 = positive [web:67][web:68]
- Size: 10,000 reviews (balanced classes). [web:67][web:68]

## Method

1. Load the CSV with pandas. [web:53]
2. Split data into train (80%) and test (20%) sets using `train_test_split` with stratification. [web:46]
3. Convert text to TF-IDF features using `TfidfVectorizer`. [web:46]
4. Train a logistic regression classifier on the training data. [web:46]
5. Evaluate on the test set using accuracy, classification report, and confusion matrix. [web:46][web:53]

## How to run

1. Install dependencies: pip install numpy pandas scikit-learn matplotlib


2. Ensure the dataset is located at: data/reviews.csv


3. Run the baseline experiment: python sentiment_baseline.py



4. Run the unigram experiment:python sentiment_unigrams.py



## Results

- Experiment 1 (1–2-gram TF-IDF + Logistic Regression): accuracy ≈ 0.8635 on test set.  
- Experiment 2 (1-gram TF-IDF + Logistic Regression): accuracy ≈ <PUT_NEW_ACCURACY_HERE> on test set.

See `experiments.md` for more details.

