from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Define request model
class TextInput(BaseModel):
    text: str

# Load TF-IDF model and vectorizer
try:
    tfidf_model = joblib.load("sentiment_analysis/tfidf_sentiment_model.pkl")
    tfidf_vectorizer = joblib.load("sentiment_analysis/tfidf_vectorizer.pkl")
    tfidf_available = True
except FileNotFoundError:
    tfidf_available = False
    print("Warning: TF-IDF model files not found")

# Load Embeddings model and embedder
try:
    embeddings_model = joblib.load("sentiment_analysis/embedding_sentiment_model.pkl")
    embeddings_embedder = joblib.load("sentiment_analysis/embedding_embedder.pkl")
    embeddings_available = True
except FileNotFoundError:
    embeddings_available = False
    print("Warning: Embeddings model files not found")


@app.get("/")
def root():
    """Root endpoint providing API information"""
    return {
        "title": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "tfidf": "/analyze/tfidf - Sentiment analysis using TF-IDF approach",
            "embeddings": "/analyze/embeddings - Sentiment analysis using Embeddings approach",
            "health": "/health - API health check"
        },
        "available_models": {
            "tfidf": tfidf_available,
            "embeddings": embeddings_available
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tfidf_model_available": tfidf_available,
        "embeddings_model_available": embeddings_available
    }


@app.post("/analyze/tfidf")
def analyze_tfidf(request: TextInput):
    """
    Analyze sentiment using TF-IDF + Logistic Regression
    
    Args:
        text: Customer text to analyze
        
    Returns:
        Sentiment prediction and confidence
    """
    if not tfidf_available:
        raise HTTPException(
            status_code=503,
            detail="TF-IDF model is not available. Please train the model first using train_sentiment_tfidf.py"
        )
    
    try:
        # Transform text using TF-IDF vectorizer
        X = tfidf_vectorizer.transform([request.text])
        
        # Get prediction
        prediction = tfidf_model.predict(X)[0]
        
        # Get probability scores
        probabilities = tfidf_model.predict_proba(X)[0]
        
        return {
            "method": "TF-IDF",
            "text": request.text,
            "sentiment": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing text: {str(e)}")


@app.post("/analyze/embeddings")
def analyze_embeddings(request: TextInput):
    """
    Analyze sentiment using SentenceTransformer Embeddings + Logistic Regression
    
    Args:
        text: Customer text to analyze
        
    Returns:
        Sentiment prediction and confidence
    """
    if not embeddings_available:
        raise HTTPException(
            status_code=503,
            detail="Embeddings model is not available. Please train the model first using train_sentiment_embeddings.py"
        )
    
    try:
        # Generate embeddings
        embedding = embeddings_embedder.encode([request.text])
        
        # Get prediction
        prediction = embeddings_model.predict(embedding)[0]
        
        # Get probability scores
        probabilities = embeddings_model.predict_proba(embedding)[0]
        
        return {
            "method": "Embeddings (SentenceTransformer)",
            "text": request.text,
            "sentiment": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing text: {str(e)}")


@app.post("/analyze/compare")
def analyze_compare(request: TextInput):
    """
    Compare sentiment predictions from both approaches
    
    Args:
        text: Customer text to analyze
        
    Returns:
        Predictions from both TF-IDF and Embeddings models
    """
    results = {
        "text": request.text,
        "tfidf": None,
        "embeddings": None
    }
    
    # Get TF-IDF prediction
    if tfidf_available:
        try:
            X = tfidf_vectorizer.transform([request.text])
            prediction = tfidf_model.predict(X)[0]
            probabilities = tfidf_model.predict_proba(X)[0]
            results["tfidf"] = {
                "sentiment": prediction,
                "confidence": float(max(probabilities))
            }
        except Exception as e:
            results["tfidf"] = {"error": str(e)}
    
    # Get Embeddings prediction
    if embeddings_available:
        try:
            embedding = embeddings_embedder.encode([request.text])
            prediction = embeddings_model.predict(embedding)[0]
            probabilities = embeddings_model.predict_proba(embedding)[0]
            results["embeddings"] = {
                "sentiment": prediction,
                "confidence": float(max(probabilities))
            }
        except Exception as e:
            results["embeddings"] = {"error": str(e)}
    
    if not tfidf_available and not embeddings_available:
        raise HTTPException(
            status_code=503,
            detail="No models available. Please train models first."
        )
    
    return results