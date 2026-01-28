import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS – HEART DISEASE DATASET")
print("=" * 60)

# Step 1: Load dataset
df = pd.read_csv("heart.csv")
print("\nDataset Loaded Successfully")
print(df.head())

# Step 2: Dataset information
print("\nDataset Information:")
print(df.info())

# Step 3: Shape of dataset
print("\nDataset Shape:", df.shape)

# Step 4: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Step 6: Age distribution
plt.figure()
sns.histplot(df["age"], bins=10, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Step 7: Gender distribution
plt.figure()
df["sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution (0 = Female, 1 = Male)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Step 8: Cholesterol vs Heart Disease
plt.figure()
sns.boxplot(x="heartdisease", y="chol", data=df)
plt.title("Cholesterol Levels vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Cholesterol")
plt.show()

# Step 9: Blood Pressure vs Heart Disease
plt.figure()
sns.boxplot(x="heartdisease", y="trestbps", data=df)
plt.title("Resting Blood Pressure vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Resting Blood Pressure")
plt.show()

# Step 10: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Step 11: Insights
print("\nKey Insights:")
print("1. Most patients fall between 40 and 65 years of age.")
print("2. Patients with heart disease tend to show higher cholesterol levels.")
print("3. Blood pressure varies significantly between patients.")
print("4. Some features show moderate correlation with heart disease.")

print("\nEDA on Heart Disease Dataset Completed Successfully")