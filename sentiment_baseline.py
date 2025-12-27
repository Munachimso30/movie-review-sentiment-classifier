import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline  # NEW


def main():
    # 1. Load data
    df = pd.read_csv("data/reviews.csv")

    # 2. Show basic info
    print("First 5 rows:")
    print(df.head())

    print("\nColumns in the dataset:")
    print(df.columns)

    print("\nSentiment value counts:")
    print(df["sentiment"].value_counts())

    # 3. Define features (X) and labels (y)
    X_text = df["review"]          # text
    y = df["sentiment"]            # labels: 0/1

    # 4. Split into train and test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,      # 20% test
        random_state=42,
        stratify=y
    )

    # 5–6. Build a pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams + bigrams
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    # 7. Train the pipeline
    pipeline.fit(X_train_text, y_train)

    # 8. SAVE the trained pipeline to a file
    joblib.dump(pipeline, "model.joblib")

    # 9. Evaluate on test set
    y_pred = pipeline.predict(X_test_text)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
