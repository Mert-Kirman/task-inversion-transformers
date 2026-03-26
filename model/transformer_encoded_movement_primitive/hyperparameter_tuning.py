import sys
import os
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from dataset import ReassembleDataset
import model.transformer_encoded_movement_primitive.temp_model as temp_model
import model.utils as utils


def normalize_data(Y1, Y2, C, training_indices):
    '''Normalize Y1, Y2, and C using global min-max normalization based on training data statistics.'''
    print("Normalizing Data (Global Min-Max)...")
    
    Y1_train = Y1[training_indices]
    Y2_train = Y2[training_indices]
    C_train = C[training_indices]

    # Combine Y1 and Y2 to find the absolute physical workspace boundaries
    Y_train_combined = torch.cat([Y1_train, Y2_train], dim=0)
    
    # Calculate ONE set of min/max values for both trajectories
    Y_min_vals = torch.amin(Y_train_combined, dim=(0, 1))  
    Y_max_vals = torch.amax(Y_train_combined, dim=(0, 1))  

    C_min_val = torch.min(C_train, dim=0).values
    C_max_val = torch.max(C_train, dim=0).values

    epsilon = 1e-8
    Y1_normalized = (Y1 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2_normalized = (Y2 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    C_normalized = (C - C_min_val) / (C_max_val - C_min_val + epsilon)

    return Y1_normalized, Y2_normalized, C_normalized, Y_min_vals, Y_max_vals, C_min_val, C_max_val

# --- W&B Sweep Training Function ---
def sweep_train():
    # Initialize a new wandb run
    wandb.init()
    config = wandb.config

    # Recreate DataLoaders with the sweep's batch size
    train_inversion_loader = DataLoader(train_inversion_dataset, batch_size=config.batch_size, shuffle=True)
    train_reconstruction_loader = DataLoader(train_reconstruction_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize Model with sweep parameters
    dropout_p = [config.dropout_p_enc, config.dropout_p_dec]
    model = temp_model.TempModel(
        full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, 
        embedding_dim=16, d_model=config.d_model, nhead=config.nhead, 
        num_layers=config.num_layers, dropout_p=dropout_p
    ).to(device)

    # Initialize Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_val_inv_mse = float('inf')
    rec_iter = iter(train_reconstruction_loader)

    for epoch in tqdm(range(config.epochs), desc="Training Progress", unit="epoch"):
        model.train()
        epoch_train_loss = 0.0
        
        for inv_batch in train_inversion_loader:
            
            # Use sweep parameter for reconstruction probability
            is_reconstruction_step = torch.rand(1).item() < config.extra_pass_prob
            
            if is_reconstruction_step:
                try:
                    batch = next(rec_iter)
                except StopIteration:
                    rec_iter = iter(train_reconstruction_loader)
                    batch = next(rec_iter)
                extra_pass = True
            else:
                batch = inv_batch
                extra_pass = False

            y1_seq = batch['y1_seq'].to(device)
            y2_seq = batch['y2_seq'].to(device)
            params = batch['context'].unsqueeze(1).to(device) 
            x_tar = batch['x_tar'].to(device)
            y_tar_f = batch['y_tar_f'].to(device)
            y_tar_i = batch['y_tar_i'].to(device)

            optimizer.zero_grad()

            batch_size = y1_seq.shape[0]
            time_len = y1_seq.shape[1]
            
            # Dynamic Masking using sweep parameter
            drop_prob = torch.rand(1).item() * config.mask_drop_prob_max 
            mask1 = torch.rand(batch_size, time_len, device=device) < drop_prob
            mask2 = torch.rand(batch_size, time_len, device=device) < drop_prob
            
            output, L_F, L_I, extra_pass = model(y1_seq, y2_seq, params, x_tar, extra_pass, mask_indices_1=mask1, mask_indices_2=mask2)
            
            loss = temp_model.loss(output, y_tar_f, y_tar_i, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass, lambda2=config.lambda2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_inversion_loader)
        
        # --- Validation ---
        if (epoch + 1) % 50 == 0:
            model.eval()
            epoch_train_fwd_mse = 0.0
            epoch_train_inv_mse = 0.0
            epoch_val_fwd_mse = 0.0
            epoch_val_inv_mse = 0.0
            
            with torch.no_grad():
                # 1. Evaluate on Training Data
                for train_batch in train_reconstruction_loader: 
                    y1_seq = train_batch['y1_seq'].to(device)
                    y2_seq = train_batch['y2_seq'].to(device)
                    params = train_batch['context'].unsqueeze(1).to(device)

                    batch_size = y1_seq.shape[0]
                    time_len = y1_seq.shape[1]
                    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1).repeat(batch_size, 1, 1)
                
                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1)
                    pred_mean_f, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                    
                    epoch_train_fwd_mse += torch.nn.functional.mse_loss(pred_mean_f, y1_seq).item()
                    epoch_train_inv_mse += torch.nn.functional.mse_loss(pred_mean_i, y2_seq).item()

                # 2. Evaluate on Validation Data
                for val_batch in val_loader:
                    y1_seq = val_batch['y1_seq'].to(device)
                    y2_seq = val_batch['y2_seq'].to(device)
                    params = val_batch['context'].unsqueeze(1).to(device)

                    batch_size = y1_seq.shape[0]
                    time_len = y1_seq.shape[1]
                    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1).repeat(batch_size, 1, 1)
                    
                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1) 
                    pred_mean_f, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                    
                    epoch_val_fwd_mse += torch.nn.functional.mse_loss(pred_mean_f, y1_seq).item()
                    epoch_val_inv_mse += torch.nn.functional.mse_loss(pred_mean_i, y2_seq).item()
            
            # Calculate averages
            avg_train_fwd_mse = epoch_train_fwd_mse / len(train_reconstruction_loader)
            avg_train_inv_mse = epoch_train_inv_mse / len(train_reconstruction_loader)
            avg_val_fwd_mse = epoch_val_fwd_mse / len(val_loader)
            avg_val_inv_mse = epoch_val_inv_mse / len(val_loader)

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch,
                "composite_train_loss": avg_train_loss,
                "train_fwd_mse": avg_train_fwd_mse,
                "train_inv_mse": avg_train_inv_mse,
                "val_fwd_mse": avg_val_fwd_mse,
                "val_inv_mse": avg_val_inv_mse,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            # --- Track Best Metric for W&B ---
            if avg_val_inv_mse < best_val_inv_mse:
                best_val_inv_mse = avg_val_inv_mse
                
                # Log best metric separately in wandb so you can easily sort your sweeps on the dashboard
                wandb.run.summary["best_val_inv_mse"] = best_val_inv_mse

if __name__ == "__main__":
    seed = 42
    utils.seed_everything(seed)

    # --- DEVICE CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- LOAD DATA ---
    base_data_folder = "data/paired_trajectories_insert_place"
    full_dataset = ReassembleDataset(data_dir=base_data_folder)

    labels = full_dataset.valid_inverses
    train_idx, val_idx = train_test_split(
        range(len(labels)), 
        test_size=0.2, 
        stratify=labels
    )

    full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, train_idx)

    paired_train_idx = [i for i in train_idx if full_dataset.valid_inverses[i]]
    
    train_inversion_dataset = Subset(full_dataset, paired_train_idx) 
    train_reconstruction_dataset = Subset(full_dataset, train_idx) 
    val_dataset = Subset(full_dataset, val_idx)

    norm_stats = {
        'Y_min': Y_min_vals,
        'Y_max': Y_max_vals,
        'C_min': C_min_val,
        'C_max': C_max_val
    }

    # --- W&B SWEEP CONFIGURATION ---
    sweep_config = {
        'method': 'bayes', 
        'metric': {
            'name': 'val_inv_mse',
            'goal': 'minimize'   
        },
        'parameters': {
            # Fixed anchors (Do not sweep these)
            'epochs': {'value': 3001},
            'batch_size': {'value': 16},
            
            # Architecture Search
            'd_model': {'values': [128, 256]},
            'num_layers': {'values': [2, 4]},
            'nhead': {'values': [2, 4, 8]},
            
            # Optimization Tuning
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-4},
            'gradient_clip_norm': {'values': [3.0, 5.0, 7.0, 10.0]},
            
            # Regularization & Masking
            'dropout_p_enc': {'values': [0.0, 0.1, 0.2]},
            'dropout_p_dec': {'values': [0.0, 0.1, 0.2]},
            'mask_drop_prob_max': {'distribution': 'uniform', 'min': 0.10, 'max': 0.99},
            'extra_pass_prob': {'distribution': 'uniform', 'min': 0.10, 'max': 0.40},
            
            # Loss Function Tuning
            'lambda2': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.1}
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="temp-ablation-sweep")

    # Run the sweep agent
    wandb.agent(sweep_id, sweep_train, count=300)
