import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

print("=" * 50)
print("Training Using TF-IDF Sentiment Analysis Model...")
print("=" * 50)

# load data
data = pd.read_csv("sentiment_analysis/calls.csv")

X_text = data["text"]
y = data["label"]

# convert text to numbers
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # learn word pairs
    min_df=1,
    stop_words="english",
    sublinear_tf=True
)
X = vectorizer.fit_transform(X_text)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

# cross validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# save both
joblib.dump(model, "sentiment_analysis/tfidf_sentiment_model.pkl")
joblib.dump(vectorizer, "sentiment_analysis/tfidf_vectorizer.pkl")


# Evaluate model with classification report
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))