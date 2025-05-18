import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('laptop_prices.csv')

# Feature Engineering

# Convert Touchscreen to binary
df['Touchscreen'] = df['Touchscreen'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert IPSpanel to binary (fixed from 'Ips')
df['Ips'] = df['IPSpanel'].apply(lambda x: 1 if x == 'Yes' else 0)

# PPI calculation from ScreenResolution column
df['X_res'] = df['ScreenW']
df['Y_res'] = df['ScreenH']
df['ppi'] = (((df['X_res'] ** 2 + df['Y_res'] ** 2) ** 0.5) / df['Inches']).astype(float)

# Helper function to convert storage sizes to GB
def get_storage_size(storage_str):
    if pd.isna(storage_str):
        return 0
    storage_str = str(storage_str).upper()  # Convert to string before calling upper()
    if 'TB' in storage_str:
        return int(float(storage_str.replace('TB', '').strip()) * 1024)
    elif 'GB' in storage_str:
        return int(float(storage_str.replace('GB', '').strip()))
    else:
        return 0

# Create numeric columns for primary and secondary storage sizes
df['PrimaryStorageSize'] = df['PrimaryStorage'].apply(get_storage_size)
df['SecondaryStorageSize'] = df['SecondaryStorage'].apply(get_storage_size)

# Extract storage types as categorical features
df['PrimaryStorageType'] = df['PrimaryStorageType'].fillna('Unknown')
df['SecondaryStorageType'] = df['SecondaryStorageType'].fillna('Unknown')

# Select required columns
df = df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
         'ppi', 'CPU_company', 'PrimaryStorageSize', 'PrimaryStorageType',
         'SecondaryStorageSize', 'SecondaryStorageType', 'GPU_company', 'OS', 'Price_euros']]

# Rename columns to keep names consistent
df.rename(columns={'CPU_company': 'Cpu brand', 'GPU_company': 'Gpu brand', 'OS': 'os', 'Price_euros': 'Price'}, inplace=True)

# Encode target variable (log price)
df['Price'] = np.log(df['Price'])

# Encode categorical variables with LabelEncoder
label_encoders = {}
for col in ['Company', 'TypeName', 'Cpu brand', 'PrimaryStorageType', 'SecondaryStorageType', 'Gpu brand', 'os']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"Test R2 Score: {r2_score(y_test, y_pred_test):.2f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved successfully!")
