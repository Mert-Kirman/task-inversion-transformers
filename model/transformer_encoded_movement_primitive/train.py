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
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    Y_min_vals = torch.amin(Y_train_combined, dim=(0, 1))  # (d_y,)
    Y_max_vals = torch.amax(Y_train_combined, dim=(0, 1))  # (d_y,)

    C_min_val = torch.min(C_train, dim=0).values
    C_max_val = torch.max(C_train, dim=0).values

    epsilon = 1e-8
    # Apply the SHARED bounds to both Y1 and Y2
    Y1_normalized = (Y1 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2_normalized = (Y2 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    C_normalized = (C - C_min_val) / (C_max_val - C_min_val + epsilon)

    # Return the shared bounds
    return Y1_normalized, Y2_normalized, C_normalized, Y_min_vals, Y_max_vals, C_min_val, C_max_val

def save_training_configs(save_folder, details_dict):
    '''Save training configurations (epoch count, batch size, input dimensions, model name etc) in a txt file for future reference'''
    details_path = os.path.join(save_folder, 'training_configs.txt')
    with open(details_path, 'w') as f:
        for key, value in details_dict.items():
            f.write(f"{key}: {value}\n")

def train(model, optimizer, scheduler, EPOCHS, train_inversion_loader, train_reconstruction_loader, val_loader, d_y1, d_y2, d_param, save_folder, device, norm_stats):
    sys.stdout = open(os.path.join(save_folder, 'train_log.txt'), 'w')

    composite_loss_list = []
    train_fwd_mse_list = []
    train_inv_mse_list = []
    val_fwd_mse_list = []
    val_inv_mse_list = []
    best_val_inv_mse = float('inf')

    # Create an iterator for the reconstruction data
    rec_iter = iter(train_reconstruction_loader)

    for epoch in tqdm(range(EPOCHS), desc="Training Progress", unit="epoch"):
        model.train()
        epoch_train_loss = 0.0
        
        # We loop over the paired (inversion) data to guarantee we see it all evenly
        for inv_batch in train_inversion_loader:
            
            # Coin flip for THIS SPECIFIC BATCH
            is_reconstruction_step = torch.rand(1).item() < 0.20
            
            if is_reconstruction_step:
                try:
                    batch = next(rec_iter)
                except StopIteration:
                    # Reset the iterator if it runs out
                    rec_iter = iter(train_reconstruction_loader)
                    batch = next(rec_iter)
                extra_pass = True
            else:
                batch = inv_batch
                extra_pass = False

            # Move data to device
            y1_seq = batch['y1_seq'].to(device)
            y2_seq = batch['y2_seq'].to(device)
            params = batch['context'].unsqueeze(1).to(device) 
            x_tar = batch['x_tar'].to(device)
            y_tar_f = batch['y_tar_f'].to(device)
            y_tar_i = batch['y_tar_i'].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output, L_F, L_I, extra_pass = model(y1_seq, y2_seq, params, x_tar, extra_pass)
            
            # Loss calculation
            loss = temp_model.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_inversion_loader)
        composite_loss_list.append(avg_train_loss)
        
        # --- Validation ---
        if (epoch + 1) % 50 == 0:
            model.eval()
            epoch_train_fwd_mse = 0.0
            epoch_train_inv_mse = 0.0
            epoch_val_fwd_mse = 0.0
            epoch_val_inv_mse = 0.0
            
            with torch.no_grad():
                # 1. Evaluate on Training Data
                for train_batch in train_reconstruction_loader: # Use all training data
                    y1_seq = train_batch['y1_seq'].to(device)
                    y2_seq = train_batch['y2_seq'].to(device)
                    params = train_batch['context'].unsqueeze(1).to(device)

                    # Generate all time points for the full trajectory
                    batch_size = y1_seq.shape[0]
                    time_len = y1_seq.shape[1]
                    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1).repeat(batch_size, 1, 1)
                
                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1) # p=1 forces L_F
                    pred_mean_f, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                    
                    # Compare full prediction against full ground truth sequences
                    epoch_train_fwd_mse += torch.nn.functional.mse_loss(pred_mean_f, y1_seq).item()
                    epoch_train_inv_mse += torch.nn.functional.mse_loss(pred_mean_i, y2_seq).item()

                # 2. Evaluate on Validation Data
                for val_batch in val_loader:
                    y1_seq = val_batch['y1_seq'].to(device)
                    y2_seq = val_batch['y2_seq'].to(device)
                    params = val_batch['context'].unsqueeze(1).to(device)

                    # Generate all time points for the full trajectory
                    batch_size = y1_seq.shape[0]
                    time_len = y1_seq.shape[1]
                    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1).repeat(batch_size, 1, 1)
                    
                    # Validation is always evaluated on Task Inversion (extra_pass = False)
                    # so we test its actual primary objective.
                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1) # p=1 means we condition on forward trajectory for inference (forces L_F to be used in decoding)
                    pred_mean_f, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                    
                    # Compare full prediction against full ground truth sequences
                    epoch_val_fwd_mse += torch.nn.functional.mse_loss(pred_mean_f, y1_seq).item()
                    epoch_val_inv_mse += torch.nn.functional.mse_loss(pred_mean_i, y2_seq).item()
            
            # Calculate averages
            avg_train_fwd_mse = epoch_train_fwd_mse / len(train_reconstruction_loader)
            avg_train_inv_mse = epoch_train_inv_mse / len(train_reconstruction_loader)
            avg_val_fwd_mse = epoch_val_fwd_mse / len(val_loader)
            avg_val_inv_mse = epoch_val_inv_mse / len(val_loader)

            train_fwd_mse_list.append(avg_train_fwd_mse)
            train_inv_mse_list.append(avg_train_inv_mse)
            val_fwd_mse_list.append(avg_val_fwd_mse)
            val_inv_mse_list.append(avg_val_inv_mse)
            
            # Save metrics
            np.save(os.path.join(save_folder, 'composite_losses.npy'), np.array(composite_loss_list))
            
            np.save(os.path.join(save_folder, 'train_fwd_mse.npy'), np.array(train_fwd_mse_list))
            np.save(os.path.join(save_folder, 'val_fwd_mse.npy'), np.array(val_fwd_mse_list))
            
            np.save(os.path.join(save_folder, 'train_inv_mse.npy'), np.array(train_inv_mse_list))
            np.save(os.path.join(save_folder, 'val_inv_mse.npy'), np.array(val_inv_mse_list))
            
            # --- Save Best Model strictly based on Zero-Shot Inversion Performance ---
            if avg_val_inv_mse < best_val_inv_mse:
                best_val_inv_mse = avg_val_inv_mse
                tqdm.write(f"Saved model epoch {epoch}, Train Inv MSE: {avg_train_inv_mse:.6f}, Val Inv MSE: {avg_val_inv_mse:.6f}")
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'norm_stats': norm_stats,
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_folder, 'best_model.pth'))


if __name__ == "__main__":
    seed = 42
    utils.seed_everything(seed)

    # --- DEVICE CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- LOAD DATA ---
    base_data_folder = "data/paired_trajectories_insert_place"
    full_dataset = ReassembleDataset(data_dir=base_data_folder)

    # --- Split Train/Val ---
    # stratify=labels guarantees the exact same ratio of paired/unpaired in train and val
    labels = full_dataset.valid_inverses
    train_idx, val_idx = train_test_split(
        range(len(labels)), 
        test_size=0.2, 
        stratify=labels
    )

    # --- Calculate normalization stats only on the training subset (avoid data leakage) ---
    full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, train_idx)

    # --- Filter indices for the Paired task (Round pegs only) ---
    paired_train_idx = [i for i in train_idx if full_dataset.valid_inverses[i]]
    
    # Create two Subsets
    train_inversion_dataset = Subset(full_dataset, paired_train_idx) # Only paired trajectories for inversion task
    train_reconstruction_dataset = Subset(full_dataset, train_idx) # Uses ALL data
    val_dataset = Subset(full_dataset, val_idx)
    
    # Create two DataLoaders
    train_inversion_loader = DataLoader(train_inversion_dataset, batch_size=16, shuffle=True)
    train_reconstruction_loader = DataLoader(train_reconstruction_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"model/transformer_encoded_movement_primitive/save/run_{run_id}"
    os.makedirs(save_folder, exist_ok=True)

    # -- MODEL, OPTIMIZER, SCHEDULER CONFIGURATION ---
    EPOCHS = 4001
    BATCH_SIZE = 16
    learning_rate = 3e-4
    weight_decay = 1e-5
    dropout_p = [0.0, 0.0]
    
    model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

    # Save training configuration details
    training_details = {
        'model_name': 'TEMP',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_p': dropout_p,
        'optimizer': 'Adam',
        'scheduler': 'LambdaLR (1.0 until 40k, then 0.5)',
        'device': str(device),
        'd_x': full_dataset.d_x,
        'd_y1': full_dataset.d_y1,
        'd_y2': full_dataset.d_y2,
        'd_param': full_dataset.d_param,
        'time_len': full_dataset.time_len,
        'num_demonstrations': full_dataset.d_N,
        'num_training_samples': len(train_idx),
        'num_validation_samples': len(val_idx),
        'Y1_shape': full_dataset.Y1.shape,
        'Y2_shape': full_dataset.Y2.shape,
        'C_shape': full_dataset.C.shape,
        'objects_config': str(full_dataset.object_config),
        'unpaired_training': True,
        'extra_pass_probability': 0.20,
        'gradient_clip_norm': 5.0,
        'seed': seed
    }
    
    save_training_configs(save_folder, training_details)

    # Package the normalization stats to save inside the checkpoint
    norm_stats = {
        'Y_min': Y_min_vals,
        'Y_max': Y_max_vals,
        'C_min': C_min_val,
        'C_max': C_max_val
    }

    train(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        EPOCHS=EPOCHS, 
        train_inversion_loader=train_inversion_loader, 
        train_reconstruction_loader=train_reconstruction_loader, 
        val_loader=val_loader, 
        d_y1=full_dataset.d_y1, 
        d_y2=full_dataset.d_y2, 
        d_param=full_dataset.d_param, 
        save_folder=save_folder, 
        device=device,
        norm_stats=norm_stats
    )
