import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import itertools

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
with PdfPages("Multivariate_Analysis_Report.pdf") as pdf:

    # ---- Heatmap of Correlation (Numerical Variables) ----
    plt.figure(figsize=(10, 7))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap of Numerical Variables", fontsize=14)
    pdf.savefig()
    plt.close()

    # ---- Grouped Bar Charts for all categorical pairs ----
    cat_pairs = list(itertools.combinations(categorical_cols, 2))

    for cat1, cat2 in cat_pairs:
        plt.figure(figsize=(10, 6))
        grouped = df.groupby([cat1, cat2])["exam_score"].mean().reset_index()
        sns.barplot(data=grouped, x=cat1, y="exam_score", hue=cat2, palette="pastel")
        plt.title(f"Mean Exam Score by {cat1} & {cat2}", fontsize=14)
        plt.xlabel(cat1)
        plt.ylabel("Mean Exam Score")
        plt.xticks(rotation=30)
        pdf.savefig()
        plt.close()

print("✅ Multivariate Analysis PDF generated with ALL categorical pairs: Multivariate_Analysis_Report.pdf")