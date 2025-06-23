# clean_titanic.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = "Titanic-Dataset.csv"  # Ensure this file is in the same folder as your script
df = pd.read_csv(file_path)

# Step 2: Explore the data
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Summary Statistics ---")
print(df.describe())

# Step 3: Handle missing values
# Fill 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Step 4: Drop irrelevant columns (like Name, Ticket)
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Step 5: Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Normalize numeric columns
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 7: Visualize outliers (optional)
# sns.boxplot(data=df[num_cols])
# plt.title("Boxplot of Age and Fare")
# plt.show()

# Step 8: Save the cleaned dataset
df.to_csv("Titanic-Cleaned.csv", index=False)
print("\n✅ Data cleaning complete. Cleaned file saved as 'Titanic-Cleaned.csv'.")
