import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

print("=" * 50)
print("Training Using Embedding-Based Sentiment Analysis Model...")
print("=" * 50)

# load dataset
data = pd.read_csv("sentiment_analysis/calls.csv")

texts = data["text"].tolist()
labels = data["label"]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert sentences → semantic vectors
X = embedder.encode(texts)

# Train classifier
model = LogisticRegression(max_iter=2000, class_weight="balanced")

scores = cross_val_score(model, X, labels, cv=5)
print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# Train final model on all data
model.fit(X, labels)

joblib.dump(model, "sentiment_analysis/embedding_sentiment_model.pkl")
joblib.dump(embedder, "sentiment_analysis/embedding_embedder.pkl")

print("Embedding model saved")