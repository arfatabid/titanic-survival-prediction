# Titanic Survival Prediction - Save and Load Model
# -------------------------------------------------
# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # used to save and load the model

# Step 2: Load data
df = pd.read_csv('train.csv')
print("Original data shape:", df.shape)

# Step 3: Clean data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Step 4: Select features (X) and target (y)
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']]
y = df['Survived']

# Step 5: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
acc = accuracy_score(y_test, y_pred)
print("\n✅ Model trained successfully!")
print("Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Save the model
joblib.dump(model, 'titanic_model.joblib')
print("\n💾 Model saved as 'titanic_model.joblib'")

# Step 10: Load the saved model (to test it)
loaded_model = joblib.load('titanic_model.joblib')
print("✅ Model loaded successfully from file!")

# Step 11: Predict for a new passenger
sample = pd.DataFrame({
    'Pclass': [2],
    'Sex': [1],
    'Age': [28],
    'Fare': [30.5],
    'Embarked': [1],
    'FamilySize': [2]
})

prediction = loaded_model.predict(sample)[0]
print("\nPrediction for new passenger:")
print("🎯 Survived" if prediction == 1 else "❌ Did not survive")
