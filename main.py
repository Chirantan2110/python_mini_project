import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
zip_path = "student.zip"
extract_dir = "student_data"

if not os.path.exists(extract_dir):
    print("Downloading dataset from UCI repository...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset downloaded and extracted.")

# Load the math class dataset (separated by semicolons)
df = pd.read_csv(f"{extract_dir}/student-mat.csv", sep=";")

# Class 0: At-risk (G3 < 10), Class 1: Satisfactory (G3 >= 10)
df['Target'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Drop original grades to avoid data leakage during prediction
X = df.drop(columns=['G1', 'G2', 'G3', 'Target'])
y = df['Target']

# Identify and fill missing values (if any)
X = X.fillna(X.mode().iloc[0])

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
evaluation_results = []
conf_matrices = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    evaluation_results.append({
        "Algorithm": name, 
        "Accuracy": acc, 
        "Precision": prec, 
        "Recall": rec, 
        "F1-Score": f1
    })
    conf_matrices[name] = cm

results_df = pd.DataFrame(evaluation_results)
print("\n--- Model Performance Comparison Table ---")
print(results_df.to_string(index=False))

# 7a. Graphical Representation: Bar chart of metrics
results_df.set_index("Algorithm").plot(kind="bar", figsize=(10, 6), colormap="viridis")
plt.title("Performance Metrics Comparison Across ML Algorithms")
plt.ylabel("Score (0.0 to 1.0)")
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# 7b. Graphical Representation: Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrices for Predictive Models', fontsize=16)

for ax, (name, cm) in zip(axes.flatten(), conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['At-Risk (0)', 'Satisfactory (1)'],
                yticklabels=['At-Risk (0)', 'Satisfactory (1)'])
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()