#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:52:36 2024

@author: ENRG AI Team
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from utils import load_data, preprocess_data, split_data

# Load and preprocess the data
filepath = "../data/data.csv"
data = load_data(filepath)
X, y = preprocess_data(data, target_col="isCherenkov", drop_cols=["isCherenkov"])
X_train, X_test, y_train, y_test = split_data(X, y)

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

# Lists to store classifier names, accuracies, and classification reports
classifier_names = []
accuracies = []
class_reports = []

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    classifier_names.append(name)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    class_report = classification_report(y_test, y_pred)
    class_reports.append(class_report)

# Write results to a text file
with open("../results/classification_results.txt", "w") as file:
    file.write("Classifier\tAccuracy\tClassification Report\n")
    for name, accuracy, class_report in zip(classifier_names, accuracies, class_reports):
        file.write(f"{name}\t{accuracy:.4f}\n{class_report}\n\n")

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.barh(classifier_names, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Accuracies')
plt.xlim(0, 1)
plt.gca().invert_yaxis()
plt.savefig('../results/model_accuracies.png', bbox_inches='tight')  # Save the plot as an image file
plt.close()


