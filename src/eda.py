import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Path
CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_students.csv")

# Visualization settings
plt.style.use("seaborn-whitegrid")
sns.set_palette("pastel")

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
df.to_csv(CLEANED_DATA_PATH, index=False)

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
    plt.show()
    
# Categorical features - bar plots
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# Bivariate Analysis
# -----------------------------
print("\n--- Bivariate Analysis ---\n")

for i,col1 in enumerate(numerical_cols):
    for col2 in numerical_cols[i+1:]:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=col1, y=col2)
        plt.title(f'{col1} vs {col2}')
        plt.show()
        
# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Categorical vs Numerical - boxplots
for cat_col in categorical_cols:
    for num_col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=cat_col, y=num_col)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()

# -------------------------------
# Multivariate Analysis
# -------------------------------
print("\n--- Multivariate Analysis ---\n") 

sns.pairplot(df, vars=numerical_cols, hue=categorical_cols[0], diag_kind='kde')
plt.suptitle("Pairwise Relationships with Hue", y=1.02)
plt.show()


#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm" , linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Scores")
plt.show()

#Compares multiple scores across categories
for cat_col in categorical_cols:
    mean_scores = df.groupby(cat_col)[numerical_cols].mean().reset_index()
    mean_scores_melted = mean_scores.melt(id_vars=cat_col, value_vars=numerical_cols,
                                            var_name="Score Type", value_name="Mean Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mean_scores_melted, x=cat_col, y="Mean Score", hue="Score Type")
    plt.title(f"Mean Scores by {cat_col}")
    plt.xticks(rotation=45)
    plt.legend(title="Score Type")
    plt.show()
    
    




