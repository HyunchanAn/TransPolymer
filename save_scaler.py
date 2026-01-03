
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load training data to fit the same scaler used during training
train_file = 'data/train_PE_I.csv'
if os.path.exists(train_file):
    train_data = pd.read_csv(train_file, header=None)
    train_data.iloc[:, 1] = pd.to_numeric(train_data.iloc[:, 1], errors='coerce')
    train_data.dropna(subset=[train_data.columns[1]], inplace=True)
    
    scaler = StandardScaler()
    scaler.fit(train_data.iloc[:, 1].values.reshape(-1, 1))
    
    # Save the scaler
    os.makedirs('ckpt', exist_ok=True)
    joblib.dump(scaler, 'ckpt/scaler.joblib')
    print("Scaler saved to ckpt/scaler.joblib")
else:
    print(f"Error: {train_file} not found.")
