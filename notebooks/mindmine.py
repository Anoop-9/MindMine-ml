import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("data/enhanced_student_habits_performance_dataset.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

if 'student_id' in df.columns:
    df.drop(columns=['student_id'], inplace=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['Study_Sleep_Interaction'] = df['study_hours_per_day'] * df['sleep_hours']
df['Social_Study_Ratio'] = df['social_media_hours'] / (df['study_hours_per_day'] + 1e-6)
df['Total_Screen_Time'] = df['social_media_hours'] + df['netflix_hours']

# Target variable
df['high_stress'] = (df['stress_level'] > 5).astype(int)

# -----------------------------
# PREPROCESSING
# -----------------------------
cols_to_exclude = ['stress_level', 'exam_score', 'high_stress']
features_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])

categorical_cols = features_df.select_dtypes(include='object').columns
X = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
y = df['high_stress']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# RESULTS
# -----------------------------
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure()
top_features.plot(kind='bar')
plt.title("Top 10 Important Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/feature_importance.png")
