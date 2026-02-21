import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load dataset
data = pd.read_csv("data.csv")

X = data[["hours"]]   # input - 2D for sklearn, even if it's just one feature
y = data["pass"]      # output - 1D for classification

# 2. Split data (training & testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model
model = LogisticRegression()

# 4. Train model
model.fit(X_train, y_train)

# 5. Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 6. Save model
joblib.dump(model, "student_model.pkl")

print("Model saved as student_model.pkl")