
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Path to the source file
source_file = 'neurips-open-polymer-prediction-2025/train.csv'

# Load data
df = pd.read_csv(source_file)

# We only need SMILES and Tg
# The project's Downstream.py expects CSV with 2 columns: SMILES, Property
tg_data = df[['SMILES', 'Tg']].dropna()

print(f"Total valid Tg data points: {len(tg_data)}")

# Split into train and test (8:2)
train_df, test_df = train_test_split(tg_data, test_size=0.2, random_state=42)

# Save to CSV (no header as the original scripts often assume no header or handle it)
# Actually, Downstream.py uses pd.read_csv(config['train_path'], header=None) in some places or handles it.
# Looking at merge_and_split.py, it saved without index and with headers (implicitly or explicitly).
# Let's check merge_and_split.py again.
# merge_and_split.py used: train_data.to_csv('data/train_PE_I.csv', index=False, header=False)
# So we follow that.

train_df.to_csv('data/train_Tg.csv', index=False, header=False)
test_df.to_csv('data/test_Tg.csv', index=False, header=False)

print("Saved data/train_Tg.csv and data/test_Tg.csv")
