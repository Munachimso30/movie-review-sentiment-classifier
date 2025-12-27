import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    df = pd.read_csv("data/reviews.csv")

    print("First 5 rows:")
    print(df.head())

    print("\nColumns in the dataset:")
    print(df.columns)

    print("\nSentiment value counts:")
    print(df["sentiment"].value_counts())

    X_text = df["review"]        
    y = df["sentiment"]          

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,     
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),  
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model = (vectorizer, clf)
    joblib.dump(model, "model.joblib")
    print("\nSaved model to model.joblib")


if __name__ == "__main__":
    main()
