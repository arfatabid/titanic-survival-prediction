# Titanic Survival Prediction 🛳️

## 📌 Project Overview
A Machine Learning project that predicts whether a passenger survived the Titanic disaster based on their age, gender, ticket class, and fare.
This project demonstrates data cleaning, model training, evaluation, and visualization.

## 🧠 Objectives
- 🧹 Clean and preprocess passenger data
- 🤖 Train a classification model (Logistic Regression / Random Forest)
- 📈 Evaluate model accuracy
- 📊 Visualize survival trends (by gender, age, class, and fare)

## 🧰 Technologies Used
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn (Machine Learning)  
- Matplotlib & Seaborn (Data Visualization)  
- Joblib (Model Saving/Loading)

## 📊 Results
- Model Accuracy: ~80%  
- Female passengers and 1st-class travelers had higher survival rates.

## 🚀 How to Run
1. Clone or download this project.  
2. Place `train.csv` in the same folder.  
3. Run:
  - 1️⃣ Step 1: Create and activate environment
      - python -m venv venv
      - venv\Scripts\activate
  - 2️⃣ Step 2: Install required libraries
      - pip install -r requirements.txt
  - 3️⃣ Step 3: Train the model
      - python train_titanic.py
  - 4️⃣ Step 4: Make predictions
      - python predict.py
  - 5️⃣ Step 5: Visualize the results
      - python visualize_titanic.py




