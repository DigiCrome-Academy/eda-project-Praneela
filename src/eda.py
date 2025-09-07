import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Path
#CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_students.csv")
FIGURES_PATH = os.path.join("reports", "figures")

os.makedirs(FIGURES_PATH, exist_ok=True)
#os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

# Visualization settings
print(plt.style.available)
#plt.style.use("seaborn-whitegrid")
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
#df.to_csv(CLEANED_DATA_PATH, index=False)

# ---------------------
# Univariate Analysis
# ---------------------

print("\n--- Univariate Analysis ---\n")

univariate_subdir = os.path.join(FIGURES_PATH, "UnivariatePlots")
os.makedirs(univariate_subdir, exist_ok=True)

# Numerical features - histograms
numerical_cols = df.select_dtypes(include=['int64','float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(univariate_subdir, f"hist_{col}.png"))
    plt.close()
    
# Categorical features - bar plots

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(univariate_subdir, f"bar_{col}.png"))
    plt.close()

# -----------------------------
# Bivariate Analysis
# -----------------------------
print("\n--- Bivariate Analysis ---\n")

bivariate_subdir = os.path.join(FIGURES_PATH, "BivariatePlots")
os.makedirs(bivariate_subdir, exist_ok=True)

for i,col1 in enumerate(numerical_cols):
    for col2 in numerical_cols[i+1:]:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=col1, y=col2)
        plt.title(f'{col1} vs {col2}')
        plt.savefig(os.path.join(bivariate_subdir, f"scatter_{col1}_{col2}.png"))
        plt.close()

        
# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(bivariate_subdir, f"heatmap_scores.png"))
plt.close()

# Categorical vs Numerical - boxplots
for cat_col in categorical_cols:
    for num_col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=cat_col, y=num_col)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(bivariate_subdir, f"boxplot_{num_col}_{cat_col}.png"))
        plt.close()

# -------------------------------
# Multivariate Analysis
# -------------------------------
print("\n--- Multivariate Analysis ---\n") 

multivariate_subdir = os.path.join(FIGURES_PATH, "MultivariatePlots")
os.makedirs(multivariate_subdir, exist_ok=True)

sns.pairplot(df, vars=numerical_cols, hue=categorical_cols[0], diag_kind='kde')
plt.suptitle("Pairwise Relationships with Hue", y=1.02)
plt.savefig(os.path.join(multivariate_subdir, f"pairplot_with_hue.png"))
plt.close()


#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm" , linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Scores")
plt.savefig(os.path.join(multivariate_subdir, f"heatmap_numerical_scores.png"))
plt.close()

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
    plt.savefig(os.path.join(multivariate_subdir, f"barplot_{cat_col}.png"))
    plt.close()    
    




