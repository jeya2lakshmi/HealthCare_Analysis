import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Create charts folder
if not os.path.exists("charts"):
    os.makedirs("charts")

# Load data
df = pd.read_csv("patients.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

# ---------------- DATA CLEANING ----------------
df["age"] = df["age"].fillna(df["age"].median())
df["gender"] = df["gender"].fillna("Unknown")
df["disease"] = df["disease"].fillna("Unknown")
df["city"] = df["city"].fillna("Unknown")

df.drop_duplicates(inplace=True)

# ---------------- FEATURE ENGINEERING ----------------
bins = [0, 18, 35, 50, 65, 100]
labels = ["0-18", "19-35", "36-50", "51-65", "65+"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

# ---------------- ANALYSIS & VISUALIZATION ----------------

# Gender Distribution
gender_count = df["gender"].value_counts()
gender_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(6, 6),
    title="Gender Distribution"
)
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/gender_distribution.png")
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("charts/age_distribution.png")
plt.show()

# Age Group Analysis
age_group_count = df["age_group"].value_counts().sort_index()
age_group_count.plot(kind="bar", figsize=(7, 4), title="Patients by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("charts/age_group_analysis.png")
plt.show()

# Top Diseases
top_diseases = df["disease"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_diseases.values, y=top_diseases.index)
plt.title("Top 10 Diseases")
plt.xlabel("Number of Patients")
plt.ylabel("Disease")
plt.tight_layout()
plt.savefig("charts/top_diseases.png")
plt.show()

# Top Cities
top_cities = df["city"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_cities.values, y=top_cities.index)
plt.title("Top 10 Cities by Patient Count")
plt.xlabel("Patients")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("charts/top_cities.png")
plt.show()

# ---------------- SUMMARY TABLES ----------------
disease_gender_summary = pd.crosstab(df["disease"], df["gender"])
age_disease_summary = pd.crosstab(df["age_group"], df["disease"])

# Export results
df.to_csv("cleaned_patient_data.csv", index=False)
disease_gender_summary.to_csv("disease_gender_summary.csv")
age_disease_summary.to_csv("age_disease_summary.csv")

print("\nAnalysis completed. Files saved successfully.")
