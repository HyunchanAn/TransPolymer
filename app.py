
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import yaml
from rdkit import Chem
from rdkit.Chem import Draw
from PolymerSmilesTokenization import PolymerSmilesTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from copy import deepcopy
import os

# Page Config
st.set_page_config(page_title="TransPolymer Predictor", layout="wide")

# --- Internationalization (i18n) ---
LANGUAGES = {
    "English": {
        "title": "🧪 TransPolymer: Multi-Property Predictor",
        "description": "Predict 5+ polymer properties simultaneously and visualize attention mechanisms.",
        "input_header": "Input SMILES",
        "input_label": "Enter Polymer SMILES:",
        "predict_btn": "Predict All Properties",
        "loading": "Loading model and assets...",
        "success_load": "Model loaded successfully!",
        "invalid_smiles": "❌ Invalid SMILES string. Please check the structure.",
        "analyzing": "Analyzing...",
        "metric_label": "Property Predictions",
        "attn_header": "Attention Map Visualization",
        "select_layer": "Select Layer",
        "select_head": "Select Head",
        "property_names": {
            "Tg": "Glass Transition (Tg) [°C]",
            "FFV": "Free Volume (FFV)",
            "Tc": "Thermal Cond (Tc) [W/mK]",
            "Density": "Density [g/cm³]",
            "Rg": "Radius of Gyration (Rg) [Å]",
            "Conductivity": "Ionic Conductivity [S/cm]"
        },
        "footer": "Powered by TransPolymer 🧬 | Deep Learning based Polymer Informatics"
    },
    "한국어": {
        "title": "🧪 TransPolymer: 통합 물성 예측기",
        "description": "SMILES 구조로부터 5가지 핵심 물성을 동시에 예측하고 AI의 분석 포인트를 확인합니다.",
        "input_header": "SMILES 입력",
        "input_label": "고분자 SMILES를 입력하세요:",
        "predict_btn": "모든 물성 예측하기",
        "loading": "모델 및 자산을 로드하는 중...",
        "success_load": "모델이 성공적으로 로드되었습니다!",
        "invalid_smiles": "❌ 유효하지 않은 SMILES 문자열입니다. 구조를 확인해주세요.",
        "analyzing": "분석 중...",
        "metric_label": "예측된 물성 결과",
        "attn_header": "Attention Map 시각화",
        "select_layer": "레이어 선택",
        "select_head": "헤드 선택",
        "property_names": {
            "Tg": "유리 전이 온도 (Tg) [°C]",
            "FFV": "자유 부피비 (FFV)",
            "Tc": "열전도도 (Tc) [W/mK]",
            "Density": "밀도 (Density) [g/cm³]",
            "Rg": "회전 반경 (Rg) [Å]",
            "Conductivity": "이온 전도도 (Ionic Cond.) [S/cm]"
        },
        "footer": "TransPolymer 🧬 | 딥러닝 기반 고분자 물성 예측 시스템"
    }
}

# Language Selection
st.sidebar.title("Language / 언어")
lang_choice = st.sidebar.radio("Select Language", options=["English", "한국어"])
txt = LANGUAGES[lang_choice]

st.title(txt["title"])
st.markdown(txt["description"])

# --- Model Definition ---
class DownstreamRegression(nn.Module):
    def __init__(self, num_outputs=1, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        # Load roberta config
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
        output = self.Regressor(logits)
        return output

# --- Cache Loaders ---
@st.cache_resource
def load_assets():
    # Use absolute paths for Windows stability
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, "configs", "config_finetune.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {'blocksize': 128}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=base_config['blocksize'])
    
    # 1. Load Multi-task Model
    model_multi = None
    scaler_multi = None
    tcols_multi = []
    ckpt_multi_path = os.path.join(curr_dir, 'ckpt', 'model_multi_best.pt')
    scaler_multi_path = os.path.join(curr_dir, 'ckpt', 'scaler_multi.joblib')
    
    if os.path.exists(ckpt_multi_path):
        checkpoint_multi = torch.load(ckpt_multi_path, map_location=device)
        tcols_multi = checkpoint_multi.get('target_cols', ['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
        model_multi = DownstreamRegression(num_outputs=len(tcols_multi)).to(device)
        model_multi.load_state_dict(checkpoint_multi['model'])
        model_multi = model_multi.double().eval()
        scaler_multi = joblib.load(scaler_multi_path)

    # 2. Load Conductivity Model
    model_cond = None
    scaler_cond = None
    ckpt_cond_path = os.path.join(curr_dir, 'ckpt', 'model_conductivity_best.pt')
    scaler_cond_path = os.path.join(curr_dir, 'ckpt', 'scaler_conductivity.joblib')
    
    if os.path.exists(ckpt_cond_path):
        checkpoint_cond = torch.load(ckpt_cond_path, map_location=device)
        model_cond = DownstreamRegression(num_outputs=1).to(device)
        model_cond.load_state_dict(checkpoint_cond['model'])
        model_cond = model_cond.double().eval()
        scaler_cond = joblib.load(scaler_cond_path)
        
    return {
        'model_multi': model_multi,
        'model_cond': model_cond,
        'tokenizer': tokenizer,
        'scaler_multi': scaler_multi,
        'scaler_cond': scaler_cond,
        'device': device,
        'config': base_config,
        'tcols_multi': tcols_multi
    }

# --- Main Logic ---
try:
    with st.spinner(txt["loading"]):
        assets = load_assets()
        model_multi = assets['model_multi']
        model_cond = assets['model_cond']
        tokenizer = assets['tokenizer']
        scaler_multi = assets['scaler_multi']
        scaler_cond = assets['scaler_cond']
        device = assets['device']
        config = assets['config']
        tcols_multi = assets['tcols_multi']
    st.success(txt["success_load"])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI Sidebar/Inputs ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(txt["input_header"])
    smiles_input = st.text_area(txt["input_label"], "CC(C)C1=CC=C(C=C1)C=C", height=100)
    
    predict_btn = st.button(txt["predict_btn"], type="primary")

    if predict_btn:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error(txt["invalid_smiles"])
            if 'results' in st.session_state:
                del st.session_state['results']
        else:
            with st.spinner(txt["analyzing"]):
                encoding = tokenizer(
                    str(smiles_input),
                    add_special_tokens=True,
                    max_length=config['blocksize'],
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                final_preds = []
                final_tcols = []
                
                with torch.no_grad():
                    # 1. Multi-task prediction
                    if model_multi:
                        pred_raw_multi = model_multi(input_ids, attention_mask)
                        pred_inv_multi = scaler_multi.inverse_transform(pred_raw_multi.cpu().numpy())[0]
                        final_preds.extend(pred_inv_multi)
                        final_tcols.extend(tcols_multi)
                    
                    # 2. Conductivity prediction
                    if model_cond:
                        pred_raw_cond = model_cond(input_ids, attention_mask)
                        pred_inv_cond = scaler_cond.inverse_transform(pred_raw_cond.cpu().numpy())[0]
                        final_preds.extend(pred_inv_cond)
                        final_tcols.append('Conductivity')
                    
                    # Get attention from the first available model for visualization
                    rep_model = model_multi if model_multi else model_cond
                    outputs = rep_model.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                    attentions = outputs.attentions
            
            # Store everything needed for display in session state
            st.session_state['results'] = {
                'prediction': final_preds,
                'attentions': attentions,
                'input_ids': input_ids,
                'smiles': smiles_input,
                'target_cols': final_tcols
            }

    # --- Persistent Display Block ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        # Show warning if input SMILES has changed since last prediction
        if res['smiles'] != smiles_input:
            st.warning("⚠️ Input SMILES has changed. Click 'Predict' to update results.")
        
        mol = Chem.MolFromSmiles(res['smiles'])
        if mol:
            st.image(Draw.MolToImage(mol, size=(400, 300)), caption="Molecular Structure")
        
        st.subheader(txt["metric_label"])
        preds = res['prediction']
        tcols = res['target_cols']
        
        # Grid for metrics
        m_cols = st.columns(3)
        for i, col_name in enumerate(tcols):
            display_name = txt["property_names"].get(col_name, col_name)
            val = preds[i]
            m_cols[i % 3].metric(display_name, f"{val:.4f}")


# --- Attention Visualization ---
if 'results' in st.session_state:
    res = st.session_state['results']
    prediction = res['prediction']
    attentions = res['attentions']
    input_ids = res['input_ids']
    
    with col2:
        st.subheader(txt["attn_header"])
        layer_to_show = st.slider(txt["select_layer"], 1, 6, 6) - 1
        head_to_show = st.selectbox(txt["select_head"], list(range(1, 13)), index=0) - 1
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        valid_indices = [i for i, t in enumerate(tokens) if t != '<pad>']
        valid_tokens = [tokens[i] for i in valid_indices]
        
        attn_matrix = attentions[layer_to_show][0, head_to_show, valid_indices, :][:, valid_indices].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn_matrix, xticklabels=valid_tokens, yticklabels=valid_tokens, ax=ax, cmap="viridis")
        plt.title(f"Layer {layer_to_show+1}, Head {head_to_show+1}")
        st.pyplot(fig)

st.divider()
st.markdown(txt["footer"])
