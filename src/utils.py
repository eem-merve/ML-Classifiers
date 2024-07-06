import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load data from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, target_col, drop_cols):
    """Preprocess the data by dropping specified columns and separating features and target."""
    X = data.drop(drop_cols, axis=1)
    y = data[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
