#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:52:36 2024

@author: ENRG AI Team
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Load the data
veri = pd.read_csv("../data/balanceddata.csv")

# Define features (X) and target (y)
X = veri.drop(["isCherenkov"], axis=1)
y = veri["isCherenkov"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers with their best parameters
model_1 = LGBMClassifier(learning_rate=0.1, max_depth=7, n_estimators=300)
model_2 = XGBClassifier(random_state=1, learning_rate=0.1, max_depth=7, subsample=0.7)
model_3 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)

# Create VotingClassifier
voting_classifier = VotingClassifier(estimators=[
    ('lgbm', model_1),
    ('xgb', model_2),
    ('rf', model_3)
], voting='soft')

# Add classifiers and their names to a list
classifiers = [model_1, model_2, model_3, voting_classifier]
classifier_names = ['LGBM', 'XGBoost', 'Random Forest', 'Ensemble']

plt.figure(figsize=(12, 10))

# Calculate and plot log loss curve for each classifier
for i, clf in enumerate(classifiers, 1):
    plt.subplot(2, 2, i)
    clf.fit(X_train, y_train)
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    log_loss_x = log_loss(y_test, y_pred_probs)
    thresholds = np.linspace(0, 1, 100)
    log_losses = []
    for threshold in thresholds:
        y_pred = (y_pred_probs > threshold).astype(int)
        log_loss_value = log_loss(y_test, y_pred)
        log_losses.append(log_loss_value)

    # Plot the graph
    plt.plot(thresholds, log_losses, linestyle='-', label=classifier_names[i - 1])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Log Loss')
    plt.title(f'Log Loss Curve for ({classifier_names[i - 1]})')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Mark the minimum Log Loss value
    min_loss_index = np.argmin(log_losses)
    min_loss_threshold = thresholds[min_loss_index]
    min_loss_value = log_losses[min_loss_index]
    plt.scatter(min_loss_threshold, min_loss_value, color='red')
    plt.annotate(f'Local Min (Base Log Loss of : {log_loss_x:.4f}\n at Predicted Probability: {min_loss_threshold:.4f})',
                 xy=(min_loss_threshold, min_loss_value),
                 xytext=(min_loss_threshold + 0.1, min_loss_value + 5),
                 horizontalalignment='center', verticalalignment='top',
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Add text in the upper right corner
    plt.text(0.95, 0.85, 'Class 0: Scintillation - 50.0%\nClass 1: Cherenkov - 50.0%',
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
