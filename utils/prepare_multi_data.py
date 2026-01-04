
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Path to the source file
source_file = 'neurips-open-polymer-prediction-2025/train.csv'

# Load data
df = pd.read_csv(source_file)

# We need SMILES and 5 properties
properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
cols = ['SMILES'] + properties
multi_data = df[cols]

print("Original dataset shape:", multi_data.shape)
# We don't dropna here because we want to use Masked Loss. 
# But we should at least have SMILES.
multi_data = multi_data.dropna(subset=['SMILES'])

print(f"Total rows with SMILES: {len(multi_data)}")

# Split into train and test (8:2)
train_df, test_df = train_test_split(multi_data, test_size=0.2, random_state=42)

# Save to CSV with header this time to keep track of columns easily in the modified Downstream.py
train_df.to_csv('data/train_Multi.csv', index=False)
test_df.to_csv('data/test_Multi.csv', index=False)

print("Saved data/train_Multi.csv and data/test_Multi.csv")
print("Property non-null counts in training set:")
print(train_df[properties].count())
