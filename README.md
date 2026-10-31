# python_mini_project
# Student Academic Performance Prediction Using Machine Learning

## 📌 Project Overview
This machine learning project predicts whether a student is at risk of poor academic performance based on their demographic, social, and academic attributes. The project implements, evaluates, and compares four supervised classification algorithms.

## 📊 Dataset
* **Dataset Name:** Student Performance Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
* **Target Variable:** Final Grade (`G3`) converted into binary classes:
  * **Class 0 (At-Risk):** $G3 < 10$
  * **Class 1 (Satisfactory):** $G3 \ge 10$

## 🛠️ Machine Learning Algorithms Implemented
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier

## 📋 Tasks Performed
1. Data downloading & loading from UCI repository.
2. Handling missing values, one-hot encoding categorical features, and standard scaling.
3. Binary class transformation of the target variable.
4. Stratified train-test split (80% training, 20% testing).
5. Model training and evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
6. Performance comparison via tables and visual plots.

## ⚙️ Installation & Running the Code
1. Clone the repository:
   ```bash
   git clone [https://github.com/Chirantan2110/Student-Performance-Prediction.git](https://github.com/Chirantan2110/Student-Performance-Prediction.git)
   cd Student-Performance-Prediction
