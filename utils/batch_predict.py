import os
import sys
import torch
import torch.nn as nn
import yaml
import joblib
import pandas as pd
import numpy as np

# Add project root to path to find local modules
sys.path.append(os.getcwd())

from transformers import RobertaModel, RobertaConfig
from PolymerSmilesTokenization import PolymerSmilesTokenizer

# --- Model Definition (Sync with app.py) ---
class DownstreamRegression(nn.Module):
    def __init__(self, num_outputs=1, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_hidden_layers = 6
        config.num_attention_heads = 12
        self.PretrainedModel = RobertaModel(config)
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, num_outputs)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        return self.Regressor(logits)

def batch_predict(input_csv, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, "configs", "config_finetune.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=config.get('blocksize', 128))
    
    # Load Models
    model_multi = None
    scaler_multi = None
    tcols_multi = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    ckpt_multi = os.path.join(curr_dir, 'ckpt', 'model_multi_best.pt')
    if os.path.exists(ckpt_multi):
        checkpoint = torch.load(ckpt_multi, map_location=device)
        model_multi = DownstreamRegression(num_outputs=len(tcols_multi)).to(device)
        model_multi.load_state_dict(checkpoint['model'])
        model_multi.double().eval()
        scaler_multi = joblib.load(os.path.join(curr_dir, 'ckpt', 'scaler_multi.joblib'))

    model_cond = None
    scaler_cond = None
    ckpt_cond = os.path.join(curr_dir, 'ckpt', 'model_conductivity_best.pt')
    if os.path.exists(ckpt_cond):
        checkpoint = torch.load(ckpt_cond, map_location=device)
        model_cond = DownstreamRegression(num_outputs=1).to(device)
        model_cond.load_state_dict(checkpoint['model'])
        model_cond.double().eval()
        scaler_cond = joblib.load(os.path.join(curr_dir, 'ckpt', 'scaler_conductivity.joblib'))

    # Load Data
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        smiles = row['smiles']
        encoding = tokenizer(
            str(smiles),
            add_special_tokens=True,
            max_length=config.get('blocksize', 128),
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(device)
        mask = encoding["attention_mask"].to(device)
        
        row_results = {'name': row['name'], 'smiles': smiles}
        
        with torch.no_grad():
            if model_multi:
                preds = model_multi(input_ids, mask)
                preds = scaler_multi.inverse_transform(preds.cpu().numpy())[0]
                for i, col in enumerate(tcols_multi):
                    row_results[col] = preds[i]
            
            if model_cond:
                preds = model_cond(input_ids, mask)
                preds = scaler_cond.inverse_transform(preds.cpu().numpy())[0]
                row_results['Ionic_Conductivity'] = preds[0]
        
        results.append(row_results)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Batch prediction complete (with Ionic Conductivity). Saved to {output_csv}")

if __name__ == "__main__":
    batch_predict("polymer_samples_predicted.csv", "polymer_samples_predicted.csv")
