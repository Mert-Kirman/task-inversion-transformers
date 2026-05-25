import argparse
import sys
import os
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import model.validate_model as validate_model
from dataset import ReassembleDataset
from model.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNMP/TEMP/TEDP models on Reassemble/Synthetic datasets.")
    parser.add_argument("--model", type=str, required=True, choices=["cnmp", "temp_vanilla", "temp_unmasked_pooling", "tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"], help="Which model architecture to train.")
    parser.add_argument("--dataset", type=str, required=True, choices=["reassemble", "synthetic_small", "synthetic_large"], help="Which dataset to train on.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args

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

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_grad_norm(model):
    '''Calculate the total L2 norm of the gradients for all parameters in the model.'''
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_cnmp(model, optimizer, scheduler, EPOCHS, valid_inverses, demo_data, obs_max, d_x, d_y1, d_y2, d_param, time_len, training_indices, validation_indices, test_indices, save_folder, device, batch_size, gradient_clip_norm, extra_pass_prob, unpaired_traj=True):
    sys.stdout = Logger(os.path.join(save_folder, 'train_log.txt'))

    training_errors = []
    validation_errors = []
    losses = []

    d_N = len(valid_inverses)
    
    for epoch in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < extra_pass_prob:
                extra_pass = True

        # Force the sampling to happen on the CPU
        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_cnmp_model.get_training_sample(
            extra_pass, valid_inverses, validation_indices, test_indices, demo_data, 
            obs_max, d_N, d_x, d_y1, d_y2, d_param, time_len, 
            batch_size=batch_size, device="cpu"
        )

        # Transfer the fully constructed tensors to the GPU all at once
        obs = obs.to(device)
        params = params.to(device)
        mask = [m.to(device) for m in mask]
        x_tar = x_tar.to(device)
        y_tar_f = y_tar_f.to(device)
        y_tar_i = y_tar_i.to(device)
        demo_data = [d.to(device) for d in demo_data]
        
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        loss = dual_cnmp_model.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 400 == 0:
            epoch_train_error = validate_model.val_only_extra(model, training_indices, epoch, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            training_errors.append(epoch_train_error if isinstance(epoch_train_error, (int, float)) else epoch_train_error.item())

            epoch_val_error = validate_model.val_only_extra(model, validation_indices, epoch, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            validation_errors.append(epoch_val_error if isinstance(epoch_val_error, (int, float)) else epoch_val_error.item())
            
            losses.append(loss.item())

            # Save errors and losses
            np.save(f'{save_folder}/training_errors_mse.npy', np.array(training_errors))
            np.save(f'{save_folder}/validation_errors_mse.npy', np.array(validation_errors))
            np.save(f'{save_folder}/losses_log_prob.npy', np.array(losses))

            if min(validation_errors) == validation_errors[-1]:
                # Save model
                tqdm.write(f"Saved model epoch {epoch}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), f'{save_folder}/best_model.pth')

def train_temp(model, optimizer, scheduler, EPOCHS, train_inversion_loader, train_reconstruction_loader, val_loader, d_y1, d_y2, d_param, save_folder, device, norm_stats, gradient_clip_norm, extra_pass_prob, obs_max):
    sys.stdout = Logger(os.path.join(save_folder, 'train_log.txt'))

    composite_loss_list = []
    train_fwd_mse_list = []
    train_inv_mse_list = []
    val_fwd_mse_list = []
    val_inv_mse_list = []
    best_val_inv_mse = float('inf')

    grad_norm_list = []

    # Create an iterator for the reconstruction data
    rec_iter = iter(train_reconstruction_loader)

    for epoch in tqdm(range(EPOCHS), desc="Training Progress", unit="epoch"):
        model.train()
        epoch_train_loss = 0.0
        epoch_grad_norm = 0.0
        
        # We loop over the paired (inversion) data to guarantee we see it all evenly
        for inv_batch in train_inversion_loader:
            
            # Coin flip for THIS SPECIFIC BATCH
            is_reconstruction_step = torch.rand(1).item() < extra_pass_prob
            
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

            batch_size = y1_seq.shape[0]
            time_len = y1_seq.shape[1]

            # Randomly decide how many observation points to keep (between 1 and obs_max)
            num_obs_f = torch.randint(1, obs_max + 1, (1,)).item()
            num_obs_i = torch.randint(1, obs_max + 1, (1,)).item()
            
            # Start with EVERYTHING masked (True means [MASK])
            mask1 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
            mask2 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
            
            # For each item in the batch, randomly unmask 'num_obs' points
            for b in range(batch_size):
                # Pick random distinct indices to unmask (False means observe)
                obs_idx1 = torch.randperm(time_len, device=device)[:num_obs_f]
                obs_idx2 = torch.randperm(time_len, device=device)[:num_obs_i]
                
                mask1[b, obs_idx1] = False
                mask2[b, obs_idx2] = False
            
            # Forward pass
            output, L_F, L_I, extra_pass = model(y1_seq, y2_seq, params, x_tar, extra_pass, mask_indices_1=mask1, mask_indices_2=mask2)
            
            # Loss calculation
            loss = temp_model.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
            
            loss.backward()
            raw_grad_norm = get_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_grad_norm += raw_grad_norm
            
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_inversion_loader)
        composite_loss_list.append(avg_train_loss)

        avg_grad_norm = epoch_grad_norm / len(train_inversion_loader)
        grad_norm_list.append(avg_grad_norm)
        
        # --- Validation ---
        if (epoch + 1) % 20 == 0:
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

                    num_obs_f = torch.randint(1, obs_max + 1, (1,)).item()
                    num_obs_i = torch.randint(1, obs_max + 1, (1,)).item()
                    
                    mask1 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    mask2 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        obs_idx_f = torch.randperm(time_len, device=device)[:num_obs_f]
                        obs_idx_i = torch.randperm(time_len, device=device)[:num_obs_i]
                        mask1[b, obs_idx_f] = False
                        mask2[b, obs_idx_i] = False
                
                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1, mask_indices_1=mask1, mask_indices_2=mask2) # p=1 forces L_F
                    pred_mean_f, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                    
                    # Compare full prediction against full ground truth sequences
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
                    
                    num_obs_f = torch.randint(1, obs_max + 1, (1,)).item()
                    num_obs_i = torch.randint(1, obs_max + 1, (1,)).item()
                    
                    mask1 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    mask2 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        obs_idx_f = torch.randperm(time_len, device=device)[:num_obs_f]
                        obs_idx_i = torch.randperm(time_len, device=device)[:num_obs_i]
                        mask1[b, obs_idx_f] = False
                        mask2[b, obs_idx_i] = False

                    output, _, _, _ = model(y1_seq, y2_seq, params, x_full, extra_pass=False, p=1, mask_indices_1=mask1, mask_indices_2=mask2) # p=1 means we condition on forward trajectory for inference (forces L_F to be used in decoding)
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

            np.save(os.path.join(save_folder, 'grad_norms.npy'), np.array(grad_norm_list))

            tqdm.write(f"Epoch {epoch}, Train Inv MSE: {avg_train_inv_mse:.6f}, Val Inv MSE: {avg_val_inv_mse:.6f}, Grad Norm: {avg_grad_norm:.4f}")
            
            # --- Save Best Model strictly based on Zero-Shot Inversion Performance ---
            if avg_val_inv_mse < best_val_inv_mse:
                best_val_inv_mse = avg_val_inv_mse
                tqdm.write(f"*** New Best Model Saved with Val Inversion MSE: {avg_val_inv_mse:.6f} ***")
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'norm_stats': norm_stats,
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_folder, 'best_model.pth'))

def train_tedp(model, optimizer, scheduler, EPOCHS, train_inversion_loader, train_reconstruction_loader, val_loader, d_y1, d_y2, d_param, save_folder, device, norm_stats, gradient_clip_norm, extra_pass_prob, obs_max):
    sys.stdout = Logger(os.path.join(save_folder, 'train_log.txt'))

    composite_loss_list = []
    train_fwd_mse_list = []
    train_inv_mse_list = []
    val_fwd_mse_list = []
    val_inv_mse_list = []
    best_val_inv_mse = float('inf')

    grad_norm_list = []

    # Create an iterator for the reconstruction data
    rec_iter = iter(train_reconstruction_loader)

    for epoch in tqdm(range(EPOCHS), desc="Training Progress", unit="epoch"):
        model.train()
        epoch_train_loss = 0.0
        epoch_grad_norm = 0.0
        
        # We loop over the paired (inversion) data to guarantee we see it all evenly
        for inv_batch in train_inversion_loader:
            
            # Coin flip for THIS SPECIFIC BATCH
            is_reconstruction_step = torch.rand(1).item() < extra_pass_prob
            
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

            optimizer.zero_grad()

            batch_size = y1_seq.shape[0]
            time_len = y1_seq.shape[1]

            num_obs_f = torch.randint(1, obs_max + 1, (1,)).item()
            num_obs_i = torch.randint(1, obs_max + 1, (1,)).item()
            
            mask1 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
            mask2 = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
            for b in range(batch_size):
                obs_idx1 = torch.randperm(time_len, device=device)[:num_obs_f]
                obs_idx2 = torch.randperm(time_len, device=device)[:num_obs_i]
                mask1[b, obs_idx1] = False
                mask2[b, obs_idx2] = False
            
            # Forward pass: Adds noise and returns the U-Net's noise prediction
            noise_pred, noise_truth, L_F, L_I, extra_pass = model(y1_seq, y2_seq, params, extra_pass, mask_indices_1=mask1, mask_indices_2=mask2)
            
            # Diffusion Loss calculation (MSE of noise + Latent Alignment)
            loss = tedp_model.loss(noise_pred, noise_truth, L_F, L_I, extra_pass, d_y1)
            
            loss.backward()
            raw_grad_norm = get_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_grad_norm += raw_grad_norm
            
        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_inversion_loader)
        composite_loss_list.append(avg_train_loss)

        avg_grad_norm = epoch_grad_norm / len(train_inversion_loader)
        grad_norm_list.append(avg_grad_norm)
        
        # --- Validation ---
        if (epoch + 1) % 20 == 0:
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
                    time_len = y1_seq.shape[1]
                    batch_size = y1_seq.shape[0]
                    
                    # Generate the Mask
                    num_obs = torch.randint(1, obs_max + 1, (1,)).item()
                    val_mask = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        obs_idx = torch.randperm(time_len, device=device)[:num_obs]
                        val_mask[b, obs_idx] = False

                    # Use Diffusion Sampling to iteratively build the trajectories
                    pred_fwd = model.sample(y1_seq, params, mask_indices=val_mask, target_dim='y1', time_len=time_len)
                    pred_inv = model.sample(y1_seq, params, mask_indices=val_mask, target_dim='y2', time_len=time_len)
                    
                    epoch_train_fwd_mse += torch.nn.functional.mse_loss(pred_fwd, y1_seq).item()
                    epoch_train_inv_mse += torch.nn.functional.mse_loss(pred_inv, y2_seq).item()

                # 2. Evaluate on Validation Data
                for val_batch in val_loader:
                    y1_seq = val_batch['y1_seq'].to(device)
                    y2_seq = val_batch['y2_seq'].to(device)
                    params = val_batch['context'].unsqueeze(1).to(device)
                    time_len = y1_seq.shape[1]
                    batch_size = y1_seq.shape[0]
                    
                    # Generate the Mask
                    num_obs = torch.randint(1, obs_max + 1, (1,)).item()
                    val_mask = torch.ones(batch_size, time_len, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        obs_idx = torch.randperm(time_len, device=device)[:num_obs]
                        val_mask[b, obs_idx] = False

                    # Diffusion Sampling
                    pred_fwd = model.sample(y1_seq, params, mask_indices=val_mask, target_dim='y1', time_len=time_len)
                    pred_inv = model.sample(y1_seq, params, mask_indices=val_mask, target_dim='y2', time_len=time_len)
                    
                    epoch_val_fwd_mse += torch.nn.functional.mse_loss(pred_fwd, y1_seq).item()
                    epoch_val_inv_mse += torch.nn.functional.mse_loss(pred_inv, y2_seq).item()
            
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

            np.save(os.path.join(save_folder, 'grad_norms.npy'), np.array(grad_norm_list))

            tqdm.write(f"Epoch {epoch}, Train Inv MSE: {avg_train_inv_mse:.6f}, Val Inv MSE: {avg_val_inv_mse:.6f}, Grad Norm: {avg_grad_norm:.4f}")
            
            # Save Best Model strictly based on Zero-Shot Inversion Performance
            if avg_val_inv_mse < best_val_inv_mse:
                best_val_inv_mse = avg_val_inv_mse
                tqdm.write(f"*** New Best Model Saved with Val Inversion MSE: {avg_val_inv_mse:.6f} ***")
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'norm_stats': norm_stats,
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_folder, 'best_model.pth'))


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # --- DEVICE CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- LOAD DATA ---
    if args.dataset == "reassemble":
        base_data_folder = "data/paired_trajectories_insert_place"
    elif args.dataset == "synthetic_small":
        base_data_folder = "data/synthetic_trajectories"
    elif args.dataset == "synthetic_large":
        base_data_folder = "data/synthetic_trajectories_large"
    
    full_dataset = ReassembleDataset(base_data_folder)

    # --- Split Train/Val/Test ---
    # stratify=labels guarantees the exact same ratio of paired/unpaired in train and val
    labels = full_dataset.valid_inverses
    train_val_idx, test_idx = train_test_split(
        range(len(labels)), 
        test_size=0.15, 
        stratify=labels,
        random_state=args.seed
    )

    # Extract the labels for the remaining 85% so we can stratify again
    train_val_labels = [labels[i] for i in train_val_idx]

    # Split the remainder into Train (70% of total) and Val (15% of total) (0.15 / 0.85 = 0.1764)
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=0.1764, 
        stratify=train_val_labels,
        random_state=args.seed
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model == "cnmp":
        save_folder = f"model/dual_cnmp_latent_alignment/save/run_{run_id}"
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling"]:
        save_folder = f"model/transformer_encoded_movement_primitive/save/run_{run_id}"
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"]:
        save_folder = f"model/transformer_encoded_diffusion_policy/save/run_{run_id}"
    os.makedirs(save_folder, exist_ok=True)

    # Save the test indices so evaluation script knows exactly which data was held out
    np.save(os.path.join(save_folder, 'test_indices.npy'), np.array(test_idx))

    # Calculate normalization stats only on the training subset (avoid data leakage)
    full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, train_idx)

    # Filter indices for the Paired task
    paired_train_idx = [i for i in train_idx if full_dataset.valid_inverses[i]]
    
    # Create two Subsets
    train_inversion_dataset = Subset(full_dataset, paired_train_idx) # Only paired trajectories for inversion task
    train_reconstruction_dataset = Subset(full_dataset, train_idx) # Uses ALL data
    val_dataset = Subset(full_dataset, val_idx)
    
    # HYPERPARAMETER CONFIGURATION
    if args.model == "cnmp":
        learning_rate = 1e-3
        weight_decay = 1e-5
        dropout_p = [0.0, 0.0]
        gradient_clip_norm = 5.0
        extra_pass_prob = 0.20
        OBS_MAX = 10
        
        if args.dataset in ["reassemble", "synthetic_small"]:
            BATCH_SIZE = 32
            EPOCHS = 18001
        elif args.dataset == "synthetic_large":
            BATCH_SIZE = 256
            EPOCHS = 4001
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling"]:
        learning_rate = 1e-3
        weight_decay = 3.5e-5
        dropout_p = [0.1, 0.0]
        gradient_clip_norm = 3.0
        extra_pass_prob = 0.25
        OBS_MAX = 10

        if args.dataset in ["reassemble", "synthetic_small"]:
            BATCH_SIZE = 32
            EPOCHS = 1001
        elif args.dataset == "synthetic_large":
            BATCH_SIZE = 128
            EPOCHS = 401
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"]:
        learning_rate = 1e-4
        weight_decay = 3.5e-5
        dropout_p = [0.1, 0.0]
        gradient_clip_norm = 3.0
        extra_pass_prob = 0.25
        OBS_MAX = 10

        if args.dataset in ["reassemble", "synthetic_small"]:
            BATCH_SIZE = 32
            EPOCHS = 1001
        elif args.dataset == "synthetic_large":
            BATCH_SIZE = 128
            EPOCHS = 601

    # Create DataLoaders
    train_inversion_loader = DataLoader(train_inversion_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_reconstruction_loader = DataLoader(train_reconstruction_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # MODEL, OPTIMIZER, SCHEDULER CONFIGURATION
    if args.model == "cnmp":
        from model.dual_cnmp_latent_alignment import dual_cnmp_model
        model = dual_cnmp_model.DualCNMP(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
    
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling"]:
        if args.model == "temp_vanilla":
            from model.transformer_encoded_movement_primitive import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
        elif args.model == "temp_unmasked_pooling":
            from model.transformer_encoded_movement_primitive.unmasked_pooling import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
        
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"]:
        if args.model == "tedp_vanilla":
            from model.transformer_encoded_diffusion_policy import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
        elif args.model == "tedp_unmasked_pooling":
            from model.transformer_encoded_diffusion_policy.unmasked_pooling import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)
        elif args.model == "tedp_cross_attention":
            from model.transformer_encoded_diffusion_policy.cross_attention_conditioning import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param, dropout_p=dropout_p).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Save training configuration details
    training_details = {
        'dataset': args.dataset,
        'model_name': args.model,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_p': dropout_p,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'device': str(device),
        'd_x': full_dataset.d_x,
        'd_y1': full_dataset.d_y1,
        'd_y2': full_dataset.d_y2,
        'd_param': full_dataset.d_param,
        'time_len': full_dataset.time_len,
        'num_demonstrations': full_dataset.d_N,
        'num_training_samples': len(train_idx),
        'num_validation_samples': len(val_idx),
        'num_test_samples': len(test_idx),
        'Y1_shape': full_dataset.Y1.shape,
        'Y2_shape': full_dataset.Y2.shape,
        'C_shape': full_dataset.C.shape,
        'objects_config': str(full_dataset.object_config),
        'unpaired_training': True,
        'extra_pass_probability': extra_pass_prob,
        'obs_max': OBS_MAX,
        'gradient_clip_norm': gradient_clip_norm,
        'seed': args.seed
    }
    
    save_training_configs(save_folder, training_details)

    # Package the normalization stats to save inside the checkpoint
    norm_stats = {
        'Y_min': Y_min_vals,
        'Y_max': Y_max_vals,
        'C_min': C_min_val,
        'C_max': C_max_val
    }

    if args.model == "cnmp":
        for key, value in norm_stats.items():
            norm_stats[key] = value.cpu()
        np.save(os.path.join(save_folder, 'normalization_stats.npy'), norm_stats)

        train_cnmp(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            EPOCHS=EPOCHS, 
            valid_inverses=full_dataset.valid_inverses,
            demo_data=[full_dataset.X1, full_dataset.X2, full_dataset.Y1, full_dataset.Y2, full_dataset.C], 
            obs_max=OBS_MAX, 
            d_x=full_dataset.d_x, 
            d_y1=full_dataset.d_y1, 
            d_y2=full_dataset.d_y2, 
            d_param=full_dataset.d_param, 
            time_len=full_dataset.time_len, 
            training_indices=train_idx, 
            validation_indices=val_idx, 
            test_indices=test_idx, 
            save_folder=save_folder, 
            device=device, 
            batch_size=BATCH_SIZE, 
            gradient_clip_norm=gradient_clip_norm,
            extra_pass_prob=extra_pass_prob,
            unpaired_traj=True
        )
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling"]:
        train_temp(
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
            norm_stats=norm_stats,
            gradient_clip_norm=gradient_clip_norm,
            extra_pass_prob=extra_pass_prob,
            obs_max=OBS_MAX
        )
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"]:
        train_tedp(
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
            norm_stats=norm_stats,
            gradient_clip_norm=gradient_clip_norm,
            extra_pass_prob=extra_pass_prob,
            obs_max=OBS_MAX
        )
