import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('spacecraft_data.csv')

# Split the data into training and testing sets
X = data[['gravity', 'alignment', 'fuel']]
y = data['trajectory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
