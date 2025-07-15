import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("D:/Pythonn/titanic dataset EDA/titanic.csv")

# Fix column name typo
df.rename(columns={"2urvived": "survived"}, inplace=True)

# Handle missing values
df = df.dropna(subset=["Embarked"])  # Drop rows with missing Embarked
df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill missing Age with median

# Check for remaining nulls
print("Missing values:\n", df.isnull().sum())

# ---------------------- PLOT 1: Survival Count by Sex ----------------------
plt.figure(figsize=(8, 5))
sns.countplot(x="Sex", hue="survived", data=df, palette="pastel")
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# ---------------------- PLOT 2: Survival Rate by Sex ----------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="Sex", y="survived", data=df, palette="viridis")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.show()

# ---------------------- PLOT 3: Survival Rate by Passenger Class ----------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="Pclass", y="survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.show()

# ---------------------- PLOT 4: Age Distribution by Survival ----------------------
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, x="Age", hue="survived", kde=True, palette="Set2", bins=30, element="step"
)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ---------------------- PLOT 5: Fare vs Survival ----------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="survived", y="Fare", data=df, palette="Set3")
plt.title("Fare Paid by Survival Status")
plt.xlabel("Survived")
plt.ylabel("Fare")
plt.show()
