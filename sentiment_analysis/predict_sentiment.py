import joblib

model = joblib.load("sentiment_analysis/sentiment_model.pkl")
vectorizer = joblib.load("sentiment_analysis/vectorizer.pkl")

text = input("Enter customer sentence: ")

X = vectorizer.transform([text])
prediction = model.predict(X)

print("Customer mood:", prediction[0])