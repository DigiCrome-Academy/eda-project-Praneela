import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FIGURES_PATH = os.path.join("reports", "figures")

os.makedirs(FIGURES_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv("data/student_habits_performance.csv")

# Shape of dataset
print("Shape of dataset:", df.shape)

# First 5 rows
print("\nFirst 5 rows:\n",df.head())

# Info about datatypes and null values
print("\nData Info:\n")
print(df.info())

# Descriptive statistics for numeric columns
print("\nDescriptive Statistics:\n", df.describe(include='all'))

# -------------------------------
# Data Cleaning
# -------------------------------
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Rename columns if necessary
df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# Save cleaned data
#df.to_csv(CLEANED_DATA_PATH, index=False)

# ---------------------
# Univariate Analysis
# ---------------------

print("\n--- Univariate Analysis ---\n")

# Numerical features - histograms
numerical_cols = df.select_dtypes(include=['int64','float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(FIGURES_PATH, f"hist_{col}.png"))
