# visualize_titanic.py
# --------------------
# Titanic Data Visualizations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the cleaned data (or original if you want)
df = pd.read_csv('train.csv')

# Step 2: Clean a few columns (like before)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Step 3: Make graphs directory (optional)
import os
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Step 4: Plot survival by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Count by Gender')
plt.xticks([0, 1], ['Male', 'Female'])
plt.savefig('graphs/survival_by_gender.png')
plt.show()

# Step 5: Plot survival by passenger class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.savefig('graphs/survival_by_class.png')
plt.show()

# Step 6: Plot Age vs Survival
sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True)
plt.title('Age Distribution by Survival')
plt.savefig('graphs/age_vs_survival.png')
plt.show()

# Step 7: Plot Fare vs Survival
sns.boxplot(data=df, x='Survived', y='Fare')
plt.title('Fare vs Survival')
plt.savefig('graphs/fare_vs_survival.png')
plt.show()

print("\n✅ All graphs saved in the 'graphs' folder!")
