#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:52:36 2024

@author: ENRG AI Team
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from utils import load_data, preprocess_data, split_data

# Define the function to compute log loss for different thresholds
def compute_log_loss_thresholds(y_test, y_pred_probs):
    thresholds = np.linspace(0, 1, 100)
    log_losses = []
    for threshold in thresholds:
        y_pred = (y_pred_probs > threshold).astype(int)
        log_loss_value = log_loss(y_test, y_pred)
        log_losses.append(log_loss_value)
    return thresholds, log_losses

# Define the function to plot the log loss curve
def plot_log_loss_curve(thresholds, log_losses, min_loss_threshold, min_loss_value, log_loss_x):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, log_losses, linestyle='-', label='RandomForest')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Log Loss')
    plt.title('Log Loss Curve for RandomForest')
    plt.grid(True)

    # Plot the local minimum log loss and corresponding threshold
    plt.scatter(min_loss_threshold, min_loss_value, color='red') # add red point to graph
    plt.legend(loc='upper right')

    # Add arrow pointing to the local minimum
    plt.annotate(f'Local Min (Base Log Loss: {log_loss_x:.4f}\n at Predicted Probability: {min_loss_threshold:.4f})',
                 xy=(min_loss_threshold, min_loss_value),  # Position of arrow head
                 xytext=(min_loss_threshold + 0.1, min_loss_value + 5),  # Position of arrow tail and text
                 horizontalalignment='center', verticalalignment='top',
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.text(0.95, 0.85, 'Class 0: Scintillation - 50.0%\nClass 1: Cherenkov - 50.0%',
             horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
    plt.show()

if __name__ == "__main__":
    # Load and preprocess your data
    input_file = "../data/balanceddata.csv"  # Update this path
    data = load_data(input_file)
    
    drop_params = ["param1", "param2"]  # Update these parameters as needed
    target_col = "target"  # Update this target column as needed

    X, y = preprocess_data(data, drop_params, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Define and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    log_loss_x = log_loss(y_test, y_pred_probs)

    # Compute log loss for different thresholds
    thresholds, log_losses = compute_log_loss_thresholds(y_test, y_pred_probs)

    # Find the predicted probability threshold corresponding to the local minimum log loss
    min_loss_index = np.argmin(log_losses)
    min_loss_threshold = thresholds[min_loss_index]
    min_loss_value = log_losses[min_loss_index]

    # Plot the log loss curve
    plot_log_loss_curve(thresholds, log_losses, min_loss_threshold, min_loss_value, log_loss_x)
