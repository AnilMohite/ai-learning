import joblib

# Load the trained model
model = joblib.load("student_model.pkl")

# ask user input
hours = float(input("Enter hours studied: "))

prediction = model.predict([[hours]])
print("Predicted class:", prediction)
if prediction[0] == 1:
    print("Prediction: The student will pass.")
else:
    print("Prediction: The student will fail.")