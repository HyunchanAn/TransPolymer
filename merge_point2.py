import pandas as pd
import os

# Load Multi-task data
train_multi = pd.read_csv('data/train_Multi.csv')
test_multi = pd.read_csv('data/test_Multi.csv')

# Load POINT2 data
point2_data = pd.read_csv('POINT2-Dataset-polymer-property-Tg/data.csv')
# Standardize columns: POINT2 has SMILES, Tg
point2_data.columns = ['SMILES', 'Tg']

print(f"Original Multi-Task train samples: {len(train_multi)}")
print(f"POINT2 samples found: {len(point2_data)}")

# --- Preprocessing POINT2 ---
# Drop duplicates if any
point2_data.drop_duplicates(subset=['SMILES'], inplace=True)

# --- Merging Strategy ---
# We want to keep all 5 properties from multi-task. 
# For POINT2, we only have Tg. FFV, Tc, Density, Rg will be NaN.
# Our masked_mse_loss is already designed to handle NaNs.

# Combine
# We'll split POINT2 90/10 for train/test to keep consistency
p2_test = point2_data.sample(frac=0.1, random_state=42)
p2_train = point2_data.drop(p2_test.index)

# Append to original
new_train = pd.concat([train_multi, p2_train], ignore_index=True)
new_test = pd.concat([test_multi, p2_test], ignore_index=True)

# Save
os.makedirs('data/merged', exist_ok=True)
new_train.to_csv('data/merged/train_Multi_POINT2.csv', index=False)
new_test.to_csv('data/merged/test_Multi_POINT2.csv', index=False)

print(f"Merged train samples: {len(new_train)}")
print(f"Merged test samples: {len(new_test)}")
print("Files saved to data/merged/")
