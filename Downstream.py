# Add project root and utils to path to find local modules
import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "utils"))

import pandas as pd
import numpy as np
import yaml
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Downstream_Dataset, DataAugmentation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from copy import deepcopy

np.random.seed(seed=1)

"""Layer-wise learning rate decay"""

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)

"""Model"""

class DownstreamRegression(nn.Module):
    def __init__(self, num_outputs=1, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, num_outputs)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output

"""Train"""

"""Masked MSE Loss for handling NaNs in Multi-task"""
def masked_mse_loss(outputs, targets):
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    diff = outputs[mask] - targets[mask]
    return (diff**2).mean()

def train(model, optimizer, scheduler, loss_fn, train_dataloader, device):

    model.train()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prop = batch["prop"].to(device).float()
        
        if step == 0: 
            print(f"DEBUG: First batch loaded! Shape: ids={input_ids.shape}, prop={prop.shape}")
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).float()
        
        # Use Masked Loss for MTL
        loss = masked_mse_loss(outputs, prop)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, loss_fn, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch, target_cols=None):

    r2score = R2Score().to(device)
    train_loss = 0
    test_loss = 0
    model.eval()
    
    # Default target list for single property
    if target_cols is None:
        target_cols_list = ["Property"]
    else:
        target_cols_list = target_cols

    with torch.no_grad():
        train_pred_list, train_true_list = [], []
        test_pred_list, test_true_list = [], []

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            
            # Masked loss for reporting
            loss = masked_mse_loss(outputs, prop)
            train_loss += loss.item() * len(prop)
            
            # Inverse transform needs to handle multi-column
            # Reshape to 2D if single property to keep scaler happy
            out_np = outputs.cpu().numpy()
            prop_np = prop.cpu().numpy()
            if len(out_np.shape) == 1: out_np = out_np.reshape(-1, 1)
            if len(prop_np.shape) == 1: prop_np = prop_np.reshape(-1, 1)
            
            outputs_inv = torch.from_numpy(scaler.inverse_transform(out_np)).to(device)
            prop_inv = torch.from_numpy(scaler.inverse_transform(prop_np)).to(device)
            
            train_pred_list.append(outputs_inv)
            train_true_list.append(prop_inv)

        train_pred = torch.cat(train_pred_list)
        train_true = torch.cat(train_true_list)
        train_loss = train_loss / len(train_pred)
        
        # Calculate R2 per property
        r2_train_per_prop = []
        for i in range(len(target_cols_list)):
            mask = ~torch.isnan(train_true[:, i])
            if mask.any():
                r2 = r2score(train_pred[mask, i], train_true[mask, i]).item()
                r2_train_per_prop.append(max(0, r2)) # Clamp to 0 for stability
            else:
                r2_train_per_prop.append(0.0)
        
        r2_train = np.mean(r2_train_per_prop)
        print(f"train RMSE = {np.sqrt(train_loss):.4f}, train avg R2 = {r2_train:.4f}")

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            
            loss = masked_mse_loss(outputs, prop)
            test_loss += loss.item() * len(prop)
            
            out_np = outputs.cpu().numpy()
            prop_np = prop.cpu().numpy()
            if len(out_np.shape) == 1: out_np = out_np.reshape(-1, 1)
            if len(prop_np.shape) == 1: prop_np = prop_np.reshape(-1, 1)
            
            outputs_inv = torch.from_numpy(scaler.inverse_transform(out_np)).to(device)
            prop_inv = torch.from_numpy(scaler.inverse_transform(prop_np)).to(device)
            
            test_pred_list.append(outputs_inv)
            test_true_list.append(prop_inv)

        test_pred = torch.cat(test_pred_list)
        test_true = torch.cat(test_true_list)
        test_loss = test_loss / len(test_pred)
        
        r2_test_per_prop = []
        print("\nProperty Metrics:")
        for i, col in enumerate(target_cols_list):
            mask = ~torch.isnan(test_true[:, i])
            if mask.any():
                r2 = r2score(test_pred[mask, i], test_true[mask, i]).item()
                mse = torch.mean((test_pred[mask, i] - test_true[mask, i])**2).item()
                print(f"  - {col}: R2={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
                r2_test_per_prop.append(r2)
            else:
                r2_test_per_prop.append(0.0)
        
        r2_test = np.mean(r2_test_per_prop)
        print(f"test RMSE = {np.sqrt(test_loss):.4f}, test avg R2 = {r2_test:.4f}\n")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("r^2/train", r2_train, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("r^2/test", r2_test, epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch, 'target_cols': target_cols}
    torch.save(state, finetune_config['save_path'])

    return train_loss, test_loss, r2_train, r2_test

    """

    if r2_test > best_test_r2:
        best_train_r2 = r2_train
        best_test_r2 = r2_test
        train_loss_best = train_loss
        test_loss_best = test_loss
        count = 0
    else:
        count += 1

    if r2_test > best_r2:
        best_r2 = r2_test
        torch.save(state, finetune_config['best_model_path'])         # save the best model

    if count >= finetune_config['tolerance']:
        print("Early stop")
        if best_test_r2 == 0:
            print("Poor performance with negative r^2")
            return None
        else:
            return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2

    return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2
    """

def main(finetune_config):

    """Tokenizer"""
    if finetune_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(finetune_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    best_r2 = 0.0           # monitor the best r^2 in the run

    """Data"""
    if finetune_config['CV_flag']:
        print("Start Cross Validation")
        data = pd.read_csv(finetune_config['train_file'])
        """K-fold"""
        splits = KFold(n_splits=finetune_config['k'], shuffle=True,
                       random_state=1)  # k=1 for train-test split and k=5 for cross validation
        train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = [], [], [], []     # monitor the best metrics in each fold
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data.shape[0]))):
            print('Fold {}'.format(fold + 1))

            train_data = data.loc[train_idx, :].reset_index(drop=True)
            test_data = data.loc[val_idx, :].reset_index(drop=True)

            if finetune_config['aug_flag']:
                print("Data Augamentation")
                DataAug = DataAugmentation(finetune_config['aug_indicator'])
                train_data = DataAug.smiles_augmentation(train_data)
                if finetune_config['aug_special_flag']:
                    train_data = DataAug.smiles_augmentation_2(train_data)
                    train_data = DataAug.combine_smiles(train_data)
                    test_data = DataAug.combine_smiles(test_data)
                train_data = DataAug.combine_columns(train_data)
                test_data = DataAug.combine_columns(test_data)

            target_cols = finetune_config.get('target_cols', None)
            if target_cols:
                print(f"Target properties for MTL: {target_cols}")
                scaler = StandardScaler()
                train_data[target_cols] = scaler.fit_transform(train_data[target_cols].values)
                test_data[target_cols] = scaler.transform(test_data[target_cols].values)
                num_outputs = len(target_cols)
            else:
                scaler = StandardScaler()
                train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
                test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))
                num_outputs = 1

            train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'], target_cols=target_cols)
            test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'], target_cols=target_cols)
            train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
            test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

            """Parameters for scheduler"""
            steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
            training_steps = steps_per_epoch * finetune_config['num_epochs']
            warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

            """Train the model"""
            model = DownstreamRegression(num_outputs=num_outputs, drop_rate=finetune_config['drop_rate']).to(device)
            # model = model.double() # Use float32 for efficiency and compatibility
            loss_fn = masked_mse_loss

            if finetune_config['LLRD_flag']:
                optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
            else:
                optimizer = AdamW(
                    [
                        {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                         "weight_decay": 0.0},
                        {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                         "weight_decay": finetune_config['weight_decay']},
                    ]
                )

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
            torch.cuda.empty_cache()
            train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
            count = 0     # Keep track of how many successive non-improvement epochs
            for epoch in range(finetune_config['num_epochs']):
                print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))
                train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
                train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch, target_cols)
                if r2_test > best_test_r2:
                    best_train_r2 = r2_train
                    best_test_r2 = r2_test
                    train_loss_best = train_loss
                    test_loss_best = test_loss
                    count = 0
                else:
                    count += 1

                if r2_test > best_r2:
                    best_r2 = r2_test
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold:': fold}
                    torch.save(state, finetune_config['best_model_path'])         # save the best model

                if count >= finetune_config['tolerance']:
                    print("Early stop")
                    if best_test_r2 == 0:
                        print("Poor performance with negative r^2")
                    break

            train_loss_avg.append(np.sqrt(train_loss_best))
            test_loss_avg.append(np.sqrt(test_loss_best))
            train_r2_avg.append(best_train_r2)
            test_r2_avg.append(best_test_r2)
            writer.flush()

        """Average of metrics over all folds"""
        train_rmse = np.mean(np.array(train_loss_avg))
        test_rmse = np.mean(np.array(test_loss_avg))
        train_r2 = np.mean(np.array(train_r2_avg))
        test_r2 = np.mean(np.array(test_r2_avg))
        std_test_rmse = np.std(np.array(test_loss_avg))
        std_test_r2 = np.std(np.array(test_r2_avg))

        print("Train RMSE =", train_rmse)
        print("Test RMSE =", test_rmse)
        print("Train R^2 =", train_r2)
        print("Test R^2 =", test_r2)
        print("Standard Deviation of Test RMSE =", std_test_rmse)
        print("Standard Deviation of Test R^2 =", std_test_r2)
    else:
        print("Train Test Split")
        target_cols = finetune_config.get('target_cols', None)
        if target_cols:
            print(f"Target properties for MTL: {target_cols}")
            train_data = pd.read_csv(finetune_config['train_file'])
            test_data = pd.read_csv(finetune_config['test_file'])
            num_outputs = len(target_cols)
            
            # Numeric conversion for all targets
            for col in target_cols:
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
                test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
                
            scaler = StandardScaler()
            train_data[target_cols] = scaler.fit_transform(train_data[target_cols].values)
            test_data[target_cols] = scaler.transform(test_data[target_cols].values)
            # Save scaler for inference
            import joblib
            joblib.dump(scaler, finetune_config.get('scaler_path', 'ckpt/scaler_multi.joblib'))
        else:
            train_data = pd.read_csv(finetune_config['train_file'], header=None)
            train_data.iloc[:, 1] = pd.to_numeric(train_data.iloc[:, 1], errors='coerce')
            train_data.dropna(subset=[train_data.columns[1]], inplace=True)

            test_data = pd.read_csv(finetune_config['test_file'], header=None)
            test_data.iloc[:, 1] = pd.to_numeric(test_data.iloc[:, 1], errors='coerce')
            test_data.dropna(subset=[test_data.columns[1]], inplace=True)
            
            scaler = StandardScaler()
            train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
            test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))
            num_outputs = 1

        if finetune_config['aug_flag']:
            print("Data Augmentation - WARNING: Only supported for single property currently")
            # Skip augmentation for MTL for now to avoid complexity unless requested
            if not target_cols:
                DataAug = DataAugmentation(finetune_config['aug_indicator'])
                train_data = DataAug.smiles_augmentation(train_data)
                if finetune_config['aug_special_flag']:
                    train_data = DataAug.smiles_augmentation_2(train_data)
                    train_data = DataAug.combine_smiles(train_data)
                    test_data = DataAug.combine_smiles(test_data)
                train_data = DataAug.combine_columns(train_data)
                test_data = DataAug.combine_columns(test_data)

        train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'], target_cols=target_cols)
        test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'], target_cols=target_cols)
        train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
        test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

        """Parameters for scheduler"""
        steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
        training_steps = steps_per_epoch * finetune_config['num_epochs']
        warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

        """Train the model"""
        model = DownstreamRegression(num_outputs=num_outputs, drop_rate=finetune_config['drop_rate']).to(device)
        # model = model.double()
        loss_fn = masked_mse_loss

        if finetune_config['LLRD_flag']:
            optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                     "weight_decay": 0.0},
                    {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                     "weight_decay": finetune_config['weight_decay']},
                ]
            )

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        torch.cuda.empty_cache()
        train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
        count = 0     # Keep track of how many successive non-improvement epochs
        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1,finetune_config['num_epochs']))
            train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
            train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch, target_cols)
            if r2_test > best_test_r2:
                best_train_r2 = r2_train
                best_test_r2 = r2_test
                train_loss_best = train_loss
                test_loss_best = test_loss
                count = 0
            else:
                count += 1

            if r2_test > best_r2:
                best_r2 = r2_test
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                torch.save(state, finetune_config['best_model_path'])         # save the best model

            if count >= finetune_config['tolerance']:
                print("Early stop")
                if best_test_r2 == 0:
                    print("Poor performance with negative r^2")
                break

        writer.flush()


if __name__ == "__main__":
    # Allow specifying config file via command line
    config_file = "configs/config_finetune.yaml"
    if len(sys.argv) > 1:
        # Simple check for --config argument
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_file = sys.argv[i+1]
                break
    
    print(f"Loading config from: {config_file}")
    finetune_config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    print(finetune_config)

    """Device"""
    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if finetune_config['model_indicator'] == 'pretrain':
        print("Use the pretrained model")
        PretrainedModel = RobertaModel.from_pretrained(finetune_config['model_path'])
        tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
        PretrainedModel.config.hidden_dropout_prob = finetune_config['hidden_dropout_prob']
        PretrainedModel.config.attention_probs_dropout_prob = finetune_config['attention_probs_dropout_prob']
    else:
        print("No Pretrain")
        config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        PretrainedModel = RobertaModel(config=config)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])
    max_token_len = finetune_config['blocksize']

    """Run the main function"""
    main(finetune_config)







