import joblib

model = joblib.load("sentiment_analysis/tfidf_sentiment_model.pkl")
vectorizer = joblib.load("sentiment_analysis/tfidf_vectorizer.pkl")

while True:
    text = input("Enter customer sentence: ")

    if text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    X = vectorizer.transform([text])
    prediction = model.predict(X)

    print("Customer mood:", prediction[0])