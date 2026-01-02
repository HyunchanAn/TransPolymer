import pandas as pd
from sklearn.model_selection import train_test_split

# Read the 'Database' sheet from the excel file
df = pd.read_excel('data/original datasets/PE_I.xlsx', sheet_name='Database')

# The script expects two columns: SMILES and the property. We will use the first two columns.
df = df.iloc[:, :2]

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save without header for the downstream script
train_df.to_csv('data/train_PE_I.csv', index=False, header=False)
test_df.to_csv('data/test_PE_I.csv', index=False, header=False)

print(f'Successfully created train_PE_I.csv with {len(train_df)} rows and test_PE_I.csv with {len(test_df)} rows.')