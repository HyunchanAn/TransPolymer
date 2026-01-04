
import torch
import joblib
import os
import yaml
from Downstream import DownstreamRegression
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from rdkit import Chem

def test_inference():
    device = torch.device("cpu")
    # Load model
    checkpoint = torch.load('ckpt/model_multi_best.pt', map_location=device)
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    model = DownstreamRegression(num_outputs=len(target_cols))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load scaler
    scaler = joblib.load('ckpt/scaler_multi.joblib')
    
    # Load tokenizer
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=128)
    
    # Test SMILES
    smiles = "CC(C)C1=CC=C(C=C1)C=C"
    encoding = tokenizer(
        smiles,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    with torch.no_grad():
        pred_normalized = model(encoding['input_ids'], encoding['attention_mask'])
        prediction = scaler.inverse_transform(pred_normalized.cpu().numpy())[0]
        
    print(f"SMILES: {smiles}")
    for name, val in zip(target_cols, prediction):
        print(f"  {name}: {val:.4f}")

if __name__ == "__main__":
    if os.path.exists('ckpt/model_multi_best.pt'):
        test_inference()
    else:
        print("Model not found!")
