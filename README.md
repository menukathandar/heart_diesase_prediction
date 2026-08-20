# heart_disease_prediction
A simple machine learning project that predicts whether a patient has heart disease based on clinical features using Logistic Regression and Support Vector Machine (SVM) classifiers.

About

This project uses the heart disease dataset to build and compare two classification models. It covers the full pipeline: data cleaning, preprocessing, model training, cross-validation, and hyperparameter tuning with GridSearchCV.

Dataset
The dataset (heart_disease.csv) contains 297 patient records (after dropping missing values) with 13 input features such as age, sex, chest pain type, blood pressure, cholesterol, and more. The Diagnosis is set as the target variable.

What the code does

Data cleaning: loads the dataset, checks for missing values and drops them. It also fixes incorrect data types and checks for duplicates.
Encoding: converts categorical variables (Sex, Fasting Blood Sugar, Chest Pain Type) into numeric values.
Preprocessing: splits the data into training and test sets (90/10 split) and normalises features using MinMaxScaler.
Model training: trains a Logistic Regression model and an SVM model, then evaluates them on the test set.
Cross-validation: uses 5-fold cross-validation to get a more reliable estimate of model performance.
Hyperparameter tuning: uses GridSearchCV to find the best parameters for both models.

Results

Model	Test Accuracy	CV Accuracy	Test Accuracy (Tuned)
Logistic Regression	96.7%	83.9%	96.7%
SVM	86.7%	80.9%	93.3%

Best Logistic Regression parameters: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}

Best SVM parameters: {'C': 1, 'degree': 3, 'gamma': 'auto', 'kernel': 'linear'}
Logistic Regression performed the best overall both before and after tuning.

Tech Stack
Python
pandas
scikit-learn

How to Run
1. Clone this repository.
2. Make sure heart_disease.csv is in the project folder.
3. Install the required libraries:
    pip install pandas scikit-learn
4. Run the script:
    python heart_disease_prediction.py
    
Notes
This project was done as part of my machine learning coursework to practice data preprocessing, model evaluation, and hyperparameter tuning.