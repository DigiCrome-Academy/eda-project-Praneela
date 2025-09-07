import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("data/student_habits_performance.csv")

# Separate numerical and categorical columns (exclude IDs)
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != "student_id"]

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != "student_id"]

# ===========================
# Create PDF Report
# ===========================
with PdfPages("Univariate_Analysis_Report.pdf") as pdf:

    # ---- Numerical Features ----
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        pdf.savefig()  # Save histogram to PDF
        plt.close()

        # Boxplot for outliers
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f"Boxplot of {col}", fontsize=14)
        plt.xlabel(col)
        pdf.savefig()
        plt.close()

    # ---- Categorical Features ----
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x=df[col], palette="pastel")
        plt.title(f"Countplot of {col}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=30)

        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height()}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha="center", va="center", 
                        fontsize=9, color="black", 
                        xytext=(0, 5), 
                        textcoords="offset points")

        pdf.savefig()
        plt.close()

print("✅ Univariate Analysis PDF generated: Univariate_Analysis_Report.pdf")