
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
        "title": "🧪 TransPolymer: Polymer Property Predictor",
        "description": "This application uses a Fine-tuned Transformer model to predict the **Conductivity** of polymers based on their SMILES string.\nIt also visualizes the attention mechanism to show which parts of the chemical structure the model is focusing on.",
        "input_header": "Input SMILES",
        "input_label": "Enter Polymer SMILES:",
        "predict_btn": "Predict Property",
        "loading": "Loading model and assets...",
        "success_load": "Model loaded successfully!",
        "invalid_smiles": "❌ Invalid SMILES string. Please check the structure.",
        "analyzing": "Analyzing...",
        "metric_label": "Predicted Conductivity",
        "attn_header": "Attention Map Visualization",
        "select_layer": "Select Layer",
        "select_head": "Select Head",
        "footer": "Developed with TransPolymer - A Transformer Language Model for Polymer Property Predictions"
    },
    "한국어": {
        "title": "🧪 TransPolymer: 고분자 물성 예측기",
        "description": "이 애플리케이션은 파인튜닝된 트랜스포머 모델을 사용하여 SMILES 문자열을 기반으로 고분자의 **전도도(Conductivity)**를 예측합니다.\n또한 어텐션 메커니즘을 시각화하여 모델이 화학 구조의 어느 부분에 집중하고 있는지 보여줍니다.",
        "input_header": "SMILES 입력",
        "input_label": "고분자 SMILES를 입력하세요:",
        "predict_btn": "물성 예측",
        "loading": "모델 및 자산을 로드하는 중...",
        "success_load": "모델이 성공적으로 로드되었습니다!",
        "invalid_smiles": "❌ 유효하지 않은 SMILES 문자열입니다. 구조를 확인해주세요.",
        "analyzing": "분석 중...",
        "metric_label": "예측된 전도도",
        "attn_header": "Attention Map 시각화",
        "select_layer": "레이어 선택",
        "select_head": "헤드 선택",
        "footer": "TransPolymer로 개발됨 - 고분자 물성 예측을 위한 트랜스포머 언어 모델"
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
    def __init__(self, pretrained_model, tokenizer_len, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(pretrained_model)
        self.PretrainedModel.resize_token_embeddings(tokenizer_len)

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output

# --- Cache Loaders ---
@st.cache_resource
def load_assets():
    with open("config_finetune.yaml", "r") as f:
        finetune_config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
    base_model = RobertaModel.from_pretrained(finetune_config['model_path']).to(device)
    
    model = DownstreamRegression(base_model, len(tokenizer)).to(device)
    checkpoint = torch.load(finetune_config['best_model_path'], map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.double()
    model.eval()
    
    scaler = joblib.load('ckpt/scaler.joblib')
    return model, tokenizer, scaler, device, finetune_config

# --- Main Logic ---
try:
    with st.spinner(txt["loading"]):
        model, tokenizer, scaler, device, config = load_assets()
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
        else:
            st.image(Draw.MolToImage(mol, size=(400, 300)), caption="Molecular Structure")
            
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
                
                with torch.no_grad():
                    pred_normalized = model(input_ids, attention_mask)
                    prediction = scaler.inverse_transform(pred_normalized.cpu().numpy())[0][0]
                    outputs = model.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                    attentions = outputs.attentions
            
            st.metric(txt["metric_label"], f"{prediction:.6f}")
            st.session_state['results'] = (prediction, attentions, input_ids)

# --- Attention Visualization ---
if 'results' in st.session_state:
    prediction, attentions, input_ids = st.session_state['results']
    
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
