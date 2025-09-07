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
with PdfPages("Bivariate_Analysis_Report.pdf") as pdf:

    # ---- Scatter Plots: Numerical vs Exam Score ----
    for col in numerical_cols:
        if col != "exam_score":  # Avoid exam_score vs exam_score
            plt.figure(figsize=(7, 5))
            sns.scatterplot(x=df[col], y=df["exam_score"], hue=df["gender"], alpha=0.7)
            plt.title(f"{col} vs Exam Score", fontsize=14)
            plt.xlabel(col)
            plt.ylabel("Exam Score")
            plt.legend(title="Gender", loc="best")
            pdf.savefig()
            plt.close()

    # ---- Box Plots: Categorical vs Exam Score ----
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col], y=df["exam_score"], palette="pastel")
        plt.title(f"Exam Score by {col}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Exam Score")
        plt.xticks(rotation=30)
        pdf.savefig()
        plt.close()

print("✅ Bivariate Analysis PDF generated: Bivariate_Analysis_Report.pdf")