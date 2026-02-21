import joblib

# load trained objects
model = joblib.load("sentiment_analysis/embedding_sentiment_model.pkl")
embedder = joblib.load("sentiment_analysis/embedding_embedder.pkl")

while True:
    text = input("Enter customer sentence: ")

    # exit condition
    if text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # convert to embedding
    X = embedder.encode([text])

    # predict
    prediction = model.predict(X)[0]

    print("Predicted sentiment:", prediction)