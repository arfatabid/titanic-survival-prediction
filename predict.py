# predict.py - Predict survival using saved model
# -----------------------------------------------

import pandas as pd
import joblib

# Step 1: Load the saved model
model = joblib.load('titanic_model.joblib')
print("✅ Model loaded successfully!\n")

# Step 2: Ask user for passenger details
print("Please enter passenger details:")

# Get user input
pclass = int(input("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd): "))
sex_input = input("Sex (male/female): ").strip().lower()
sex = 0 if sex_input == 'male' else 1
age = float(input("Age (in years): "))
fare = float(input("Ticket Fare: "))
embarked_input = input("Embarked (S/C/Q): ").strip().upper()
embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked_input]
family_size = int(input("Family Size (1 if alone, otherwise SibSp + Parch + 1): "))

# Step 3: Make a DataFrame for the model
data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Fare': [fare],
    'Embarked': [embarked],
    'FamilySize': [family_size]
})

# Step 4: Predict survival
prediction = model.predict(data)[0]

# Step 5: Show result
print("\n🎯 Prediction Result:")
if prediction == 1:
    print("✅ The passenger **Survived!** 🎉")
else:
    print("❌ The passenger **Did NOT Survive.** 😢")
