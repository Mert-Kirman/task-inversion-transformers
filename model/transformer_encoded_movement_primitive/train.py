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
import model.validate_model as validate_model
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

def train(model, optimizer, scheduler, EPOCHS, train_loader, d_y1, d_y2, d_param, time_len, save_folder, device):
    sys.stdout = open(os.path.join(save_folder, 'train_log.txt'), 'w')

    training_errors = []
    validation_errors = []
    losses = []

    for epoch in tqdm(range(EPOCHS)):
        for batch in train_loader:
            
            # Move data to device
            y1_seq = batch['y1_seq'].to(device) # Shape: (batch, time_len, d_y1)
            y2_seq = batch['y2_seq'].to(device)
            params = batch['context'].unsqueeze(1).to(device) # Shape: (batch, 1, d_param)
            x_tar = batch['x_tar'].to(device) # Shape: (batch, 1, d_x)
            y_tar_f = batch['y_tar_f'].to(device) # Shape: (batch, 1, d_y1)
            y_tar_i = batch['y_tar_i'].to(device)
            valid_inverses_batch = batch['is_valid_inverse'] # Shape: (batch,)

            # Logic for extra_pass (Unpaired data)
            # If the batch contains any invalid inverses, or randomly 20% of the time
            extra_pass = False
            if not all(valid_inverses_batch) or torch.rand(1).item() < 0.20:
                extra_pass = True

            optimizer.zero_grad()
            
            # Note: The model forward signature will change when we add BERT!
            # For now, it might look something like this:
            output, L_F, L_I, extra_pass = model(y1_seq, y2_seq, params, x_tar, extra_pass)
            
            loss = temp_model.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 50 == 0:
            epoch_train_error = validate_model.val_only_extra(model, training_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            training_errors.append(epoch_train_error if isinstance(epoch_train_error, (int, float)) else epoch_train_error.item())

            epoch_val_error = validate_model.val_only_extra(model, validation_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            validation_errors.append(epoch_val_error if isinstance(epoch_val_error, (int, float)) else epoch_val_error.item())
            
            losses.append(loss.item())

            # Save errors and losses
            np.save(os.path.join(save_folder, 'training_errors_mse.npy'), np.array(training_errors))
            np.save(os.path.join(save_folder, 'validation_errors_mse.npy'), np.array(validation_errors))
            np.save(os.path.join(save_folder, 'losses_log_prob.npy'), np.array(losses))

            if min(validation_errors) == validation_errors[-1]:
                # Save model
                tqdm.write(f"Saved model epoch {epoch}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))


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

    # Calculate normalization stats only on the training subset (important to avoid data leakage!)
    full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, train_idx)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
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
    train(
        model, optimizer, scheduler, EPOCHS, 
        valid_inverses, demo_data, OBS_MAX, d_x, d_y1, d_y2, d_param, time_len,
        validation_indices, training_indices, save_folder, run_id, device, 
        batch_size=BATCH_SIZE, unpaired_traj=True
    )
