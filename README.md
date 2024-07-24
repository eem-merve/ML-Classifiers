# Comprehensive Machine Learning Model Comparison for Cherenkov and Scintillation Light Separation due to Particle Interactions

This project compares several machine learning classifiers on a Cherenkov/Scintillation dataset to determine their accuracy. This study comprehensively evaluated Machine Learning (ML) models for classifying Cherenkov and scintillation photons due to neutrino interactions. All the related parameters and their combinations such as time, energy, and PMT coordinates in the detector were carefully studied. This project can provide ML-based classification methods for separating Cherenkov and scintillation photons. This study comprehensively evaluated ML models for classifying Cherenkov and scintillation photons due to neutrino interactions. Several ML models were compared in balanced and unbalanced datasets, and their accuracies were calculated. All the related parameters and their combinations such as time, energy, and PMT coordinates in the detector were carefully studied.
# Ensemble Classification with Log Loss Curves

This project demonstrates the use of ensemble classification methods to predict the presence of Cherenkov radiation. The project uses a dataset of features to train and evaluate multiple classifiers, including LightGBM, XGBoost, and Random Forest and combines their predictions using a voting classifier. The performance of the classifiers is visualized using log loss curves.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Citiation](#citiation)


## Introduction

Cherenkov radiation is electromagnetic radiation emitted when a charged particle moves through a dielectric medium at a speed greater than the speed of light in that medium. This project aims to classify instances based on various features to determine if Cherenkov radiation is present.

The project focuses on three classifiers:
- LightGBM
- XGBoost
- Random Forest

These classifiers are combined using a soft voting classifier to improve the prediction performance.

## Dataset

The dataset used in this project are `balanceddata.csv` and `unbalanceddata.csv`. It contains various features, and the target variable is `Vector`, which indicates the presence of Cherenkov radiation.

## Dependencies

To run this project, you need the following dependencies:

- Python 3.7+
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- matplotlib

You can install the required Python packages using pip:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib
```
# Usage
git clone https://github.com/yourusername/yourrepository.git](https://github.com/eem-merve/ML-Classifiers

cd src

python log_loss_plot.py

# Results

The results of the classifiers are visualized using log loss curves. Each curve shows the log loss for different predicted probability thresholds. The script also identifies and marks the threshold that yields the minimum log loss for each classifier.

The generated plots will be saved in the results directory.

# Citation
@article{tiras2024comprehensive,
  title={Comprehensive Machine Learning Model Comparison for Cherenkov and Scintillation Light Separation due to Particle Interactions},
  author={Tiras, Emrah and Tas, Merve and Kizilkaya, Dilara and Yagiz, Muhammet Anil and Kandemir, Mustafa},
  journal={arXiv preprint arXiv:2406.09191},
  year={2024}
}
