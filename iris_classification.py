# ==========================================
# IRIS FLOWER CLASSIFICATION PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
import joblib

# ==========================================
# LOAD DATASET
# ==========================================

# Load CSV file
df = pd.read_csv("iris.csv")

# Display column names
print("\nColumn Names:\n")
print(df.columns)

# Display first 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset Information
print("\nDataset Info:\n")
print(df.info())

# Check Null Values
print("\nNull Values:\n")
print(df.isnull().sum())

# ==========================================
# DATA VISUALIZATION
# ==========================================

# Pairplot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# PREPROCESSING
# ==========================================

# Remove Id column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Encode target labels
encoder = LabelEncoder()

df["Species"] = encoder.fit_transform(df["Species"])

# Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

# ==========================================
# MODEL PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:\n")
print(accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

species_name = encoder.inverse_transform(prediction)

print("\nPredicted Species:")
print(species_name[0])

# ==========================================
# END OF PROJECT
# ==========================================