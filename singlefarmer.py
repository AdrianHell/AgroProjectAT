import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Read the Excel file
excel_file_path = 'FARMER1 All features.xlsx'  # Adjust the path if needed
excel_data = pd.read_excel(excel_file_path)

# Prepare an empty list to store predictions
predictions = []


for _, row in excel_data.iloc[13:49].iterrows():
    # Extract feature values and reshape them to 2D
    features = np.array(row.values[3:].tolist()).reshape(1, -1)
    
    # Predict using your model
    predicted_score = model.predict(features)
    predictions.append(predicted_score[0])


print(predictions)
print(len(predictions))



# List of month date strings
months = [
    "2021-03-31 12:00:00", "2021-04-30 12:00:00",
    "2021-05-31 12:00:00", "2021-06-30 12:00:00", "2021-07-31 12:00:00",
    "2021-08-31 12:00:00", "2021-09-30 12:00:00", "2021-10-31 12:00:00",
    "2021-11-30 12:00:00", "2021-12-31 12:00:00", "2022-01-31 12:00:00",
    "2022-02-28 12:00:00", "2022-03-31 12:00:00", "2022-04-30 12:00:00",
    "2022-05-31 12:00:00", "2022-06-30 12:00:00", "2022-07-31 12:00:00",
    "2022-08-31 12:00:00", "2022-09-30 12:00:00", "2022-10-31 12:00:00",
    "2022-11-30 12:00:00", "2022-12-31 12:00:00", "2023-01-31 12:00:00",
    "2023-02-28 12:00:00", "2023-03-31 12:00:00", "2023-04-30 12:00:00",
    "2023-05-31 12:00:00", "2023-06-30 12:00:00", "2023-07-31 12:00:00",
    "2023-08-31 12:00:00", "2023-09-30 12:00:00", "2023-10-31 12:00:00",
    "2023-11-30 12:00:00", "2023-12-31 12:00:00", "2024-01-31 12:00:00"
]

print(len(months))

# Create a DataFrame
df = pd.DataFrame({
    'Month': months,
    'Predictions': predictions
})

# Write to Excel, starting from row 2 (index 1)
df.to_excel('Predictions.xlsx', index=False, startrow=0)

# Load the Excel file
file_path = 'Predictions.xlsx'
df = pd.read_excel(file_path)

# Convert the 'Month' column to datetime for better plotting
df['Month'] = pd.to_datetime(df['Month'])

# Setting the plot style
sns.set(style="whitegrid")

# Creating the plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Predictions', data=df, marker='o')

# Customizing the plot
plt.title('Monthly Predictions')
plt.xlabel('Month')
plt.ylabel('Predictions')
plt.xticks(rotation=45)

# Show the plot
plt.show()



