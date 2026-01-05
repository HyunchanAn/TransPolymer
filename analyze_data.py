import pandas as pd
import numpy as np

files = ['data/train_Multi.csv', 'data/train_PE_I.csv']
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n--- {f} ---")
        print(f"Columns: {df.columns.tolist()}")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                desc = df[col].describe()
                print(f"[{col}]")
                print(f"  mean: {desc['mean']:.6f}")
                print(f"  min:  {desc['min']:.6f}")
                print(f"  max:  {desc['max']:.6f}")
                print(f"  std:  {desc['std']:.6f}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
