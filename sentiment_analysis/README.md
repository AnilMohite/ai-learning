# Sentiment Analysis Model

A machine learning project for analyzing customer sentiment from call transcripts using natural language processing.

## Project Overview

This project builds and deploys a sentiment classification model to determine customer mood/sentiment from text data. It uses TF-IDF vectorization combined with Logistic Regression for fast and interpretable predictions.

## Project Structure

```
call_model/
├── README.md                  # This file
├── calls.csv                  # Dataset with customer call texts and sentiment labels
├── train_sentiment.py         # Model training script
├── predict_sentiment.py       # Command-line prediction script
├── app.py                     # FastAPI web service
├── sentiment_model.pkl        # Trained model (generated after training)
└── vectorizer.pkl             # Text vectorizer (generated after training)
```

## Features

- **Data Processing**: TF-IDF vectorization for converting text to numerical features
- **Model**: Logistic Regression classifier for binary/multi-class sentiment prediction
- **Training**: Automated model training with 80-20 train-test split
- **Inference**: Two modes - command-line interface and REST API
- **Model Persistence**: Save and load trained models using joblib

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- joblib
- fastapi
- uvicorn (for running the API)

Install dependencies:
```bash
pip install pandas scikit-learn joblib fastapi uvicorn
```

## Data Format

The `calls.csv` file should have the following structure:

```csv
text,label
"customer statement...",positive
"another statement...",negative
```

## Usage

### 1. Train the Model

```bash
python train_sentiment.py
```

This will:
- Load data from `calls.csv`
- Vectorize text using TF-IDF
- Split data into training and testing sets
- Train a Logistic Regression model
- Display test accuracy
- Save `sentiment_model.pkl` and `vectorizer.pkl`

### 2. Make Predictions (CLI)

```bash
python predict_sentiment.py
```

Interactive mode - enter customer text to get sentiment predictions:
```
Enter customer sentence: I love your service!
Customer mood: positive
```

### 3. Run API Server

```bash
uvicorn app:app --reload
```

Server runs on `http://localhost:8000`

**API Endpoint:**
```
POST /analyze
Content-Type: application/json

{
  "text": "Your customer message here"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Great customer service!"}'
```

## Model Performance

The model accuracy is printed during training on the test set.

## Future Improvements

- Add cross-validation for more robust performance estimates
- Implement sentiment probability scores instead of just labels
- Add data preprocessing (lowercasing, removing stopwords)
- Deploy model to production (Docker, AWS, etc.)
- Add more advanced models (SVM, Neural Networks)

## License

Open source - feel free to modify and use for your project.
