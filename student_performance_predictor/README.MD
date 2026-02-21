# Student Pass/Fail Prediction Model

A machine learning model that predicts whether a student will pass or fail based on the number of hours studied using Logistic Regression.

## Overview

This project implements a binary classification model to predict student academic outcomes. It analyzes the relationship between study hours and exam performance to make predictions about future student success.

## Features

- **Binary Classification**: Predicts pass/fail outcomes
- **Simple & Effective**: Uses Logistic Regression for clear, interpretable results
- **Easy to Use**: Simple command-line interface for making predictions
- **Model Persistence**: Trained model is saved for reuse without retraining

## Dataset

The model is trained on `data.csv` which contains:
- `hours`: Number of hours studied (numeric feature)
- `pass`: Binary outcome (1 = Pass, 0 = Fail)

## Installation

### Requirements
- Python 3.7+
- pandas
- scikit-learn
- joblib

### Setup

```bash
pip install pandas scikit-learn joblib
```

## Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Load the dataset from `data.csv`
- Split data into 80% training and 20% testing sets
- Train the Logistic Regression model
- Display model accuracy
- Save the trained model as `student_model.pkl`

### 2. Make Predictions

```bash
python predict.py
```

Enter the number of hours studied when prompted, and the model will predict whether the student will pass or fail.

**Example:**
```
Enter hours studied: 5
Predicted class: [1]
Prediction: The student will pass.
```

## Project Structure

```
prediction_model/
├── data.csv          # Training dataset
├── train.py          # Model training script
├── predict.py        # Prediction script
└── README.MD         # Project documentation
```

## Model Details

- **Algorithm**: Logistic Regression
- **Input**: Hours studied (1 feature)
- **Output**: Pass (1) or Fail (0)
- **Train-Test Split**: 80-20
- **Saved Model**: `student_model.pkl`

## Notes

- The model file (`student_model.pkl`) must exist before running predictions
- Run `train.py` first to generate the model file
- Random state is set to 42 for reproducibility