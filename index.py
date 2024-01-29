import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import pickle


# Load the dataset
file_path = 'Rounded_Assessment_Scores(1).csv'  # Replace with your file path
data = pd.read_csv(file_path)
data_cleaned = data.drop(columns=[col for col in data.columns if "Unnamed" in col])

# Separate features and target variable
X = data_cleaned.drop('FINAL SCORE', axis=1)
y = data_cleaned['FINAL SCORE']

weights = [0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1]  

# Apply the weights to each feature
for i, col in enumerate(X.columns):
    X[col] = X[col] * weights[i]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                     max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train, y_train)

with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to make predictions
def predict_credit_score(features):
    """
    Predict the credit score based on input features.
    Args:
    features (list): A list of feature values in the order of the dataset columns.

    Returns:
    float: The predicted credit score.
    """
    input_data = pd.DataFrame([features], columns=X.columns)
    return model.predict(input_data)[0].item()

# Example usage
example_features = [63613.4, 515344.09, 212044.66, 50593.08, 6.760838757, 0, 10]  # Replace with your own feature values

# https://github.com/sajal-01/crop-recommend-python

# predicted_score = predict_credit_score(example_features)
# print("Predicted Credit Score:", predicted_score)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

# Convert to Percentage
r2_percentage = r2 * 100

# Print the result
print(f"R2 Score (Percentage): {r2_percentage:.2f}%")
