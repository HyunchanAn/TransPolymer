
import pandas as pd
from sklearn.model_selection import train_test_split
import re

print("Reading Excel file...")
xlsx = pd.ExcelFile('data/original datasets/PE_I.xlsx')

print("Reading Database sheet...")
df_db = pd.read_excel(xlsx, sheet_name='Database')
# Target: Conductivity [S/cm] (assuming it's the one we found earlier)
# Let's verify column names first to be safe, but based on previous output it is 'Conductivity [S/cm]' (or similar)
cond_col = [c for c in df_db.columns if 'cond' in str(c).lower()][0]
comp_col = 'Composition'
print(f"Using columns: {comp_col} (Structure ref) and {cond_col} (Property)")

print("Reading CompoundDatabase sheet...")
df_cdb = pd.read_excel(xlsx, sheet_name='CompoundDatabase')
# Target: Nickname (ref) and SMILES
nick_col = 'Nickname'
smiles_col = 'SMILES'

# Create a mapping from Nickname to SMILES
# Nickname might be int or str, normalize to str
df_cdb[nick_col] = df_cdb[nick_col].astype(str)
nickname_to_smiles = dict(zip(df_cdb[nick_col], df_cdb[smiles_col]))

print("Processing Composition column...")
def get_smiles(composition):
    # Composition format looks like '1/s1' or '1'. 
    # Assumption: The first part before '/' (if exists) is the proper polymer ID (Nickname).
    # Or maybe it's just the number.
    # Let's try to extract the first number found or the whole string before the first special char.
    if pd.isna(composition):
        return None
    s_comp = str(composition).strip()
    
    # Simple strategy: splits by '/' and takes the first part.
    # Check if '1' is in nickname_to_smiles
    parts = s_comp.split('/')
    ref = parts[0].strip()
    
    return nickname_to_smiles.get(ref, None)

df_db['SMILES'] = df_db[comp_col].apply(get_smiles)

# Filter out missing SMILES or Property
df_clean = df_db.dropna(subset=['SMILES', cond_col])
print(f"Rows after merging and cleaning: {len(df_clean)} (original: {len(df_db)})")

# Select only SMILES and Property
df_final = df_clean[['SMILES', cond_col]]

# Split
print("Splitting data...")
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

# Save
print("Saving with NO header...")
train_df.to_csv('data/train_PE_I.csv', index=False, header=False)
test_df.to_csv('data/test_PE_I.csv', index=False, header=False)

print("Done!")
