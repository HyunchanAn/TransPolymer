
import pandas as pd
df = pd.read_excel('data/original datasets/PE_I.xlsx', sheet_name='CompoundDictionary', nrows=5)
print(df.columns.tolist())
print(df.head())
