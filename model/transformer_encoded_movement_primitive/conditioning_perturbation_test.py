import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Adjust path to find model modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import ReassembleDataset
import model.transformer_encoded_movement_primitive.temp_model as temp_model
import model.model_predict as model_predict
import model.utils as utils

# ================= CONFIGURATION =================
run_id = "run_20260408_204033"
save_path = f"model/transformer_encoded_movement_primitive/save/{run_id}"
# =================================================

def normalize_data(Y1, Y2, C, Y_min_vals, Y_max_vals, C_min_val, C_max_val):
    """Normalize Y1, Y2, and C using global min-max normalization based on training data statistics."""
    epsilon = 1e-8
    Y1_normalized = (Y1 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2_normalized = (Y2 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    C_normalized = (C - C_min_val) / (C_max_val - C_min_val + epsilon)

    return Y1_normalized, Y2_normalized, C_normalized

def denormalize_data(tensor, min_val, max_val):
    """Reverts [0, 1] data back to original scale."""
    denominator = max_val - min_val
    return tensor * denominator + min_val

def condition_perturbation_test(base_data_folder, num_samples=5, device='cpu'):
    print(f"\n--- RUNNING CONDITIONING PERTURBATION TEST ({num_samples} samples) ---")

    # Load Data & Stats
    full_dataset = ReassembleDataset(data_dir=base_data_folder)
    
    checkpoint = torch.load(os.path.join(save_path, "best_model.pth"))
    norm_stats = checkpoint['norm_stats']
    Y_min_vals, Y_max_vals = norm_stats['Y_min'], norm_stats['Y_max']
    C_min_val, C_max_val = norm_stats['C_min'], norm_stats['C_max']

    # Keep a raw copy of Y2 for plotting ground truth
    Y2_raw = full_dataset.Y2.clone()

    # Normalize dataset
    full_dataset.Y1, full_dataset.Y2, full_dataset.C = normalize_data(
        full_dataset.Y1, full_dataset.Y2, full_dataset.C, 
        Y_min_vals, Y_max_vals, C_min_val, C_max_val
    )

    d_x = full_dataset.d_x
    d_y1 = full_dataset.d_y1 
    d_y2 = full_dataset.d_y2 
    d_param = full_dataset.d_param
    time_len = full_dataset.time_len

    # Load Model
    model = temp_model.TempModel(d_x, d_y1, d_y2, d_param).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # Load test indices and sample from them
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy'))
    test_idx_list = test_idx.tolist()
    
    num_to_plot = min(num_samples, len(test_idx_list))
    indices = random.sample(test_idx_list, num_to_plot)
    
    time_steps = np.linspace(0, 1, time_len)
    print(f"Evaluating indices: {indices}")

    # Set up the plot
    fig, axes = plt.subplots(num_to_plot, d_y2, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    # Condition on t=0 and t=1
    cond_step_indices = [0, -1] 
    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
    eval_mask[0, cond_step_indices] = False # False = Observed

    for row_idx, traj_idx in enumerate(indices):
        # Identify Object Type
        curr_context_norm = full_dataset.C[traj_idx]
        raw_id = denormalize_data(curr_context_norm.to(device), C_min_val.to(device), C_max_val.to(device))[-1].item()
        
        curr_obj_name = "Unknown"
        min_diff = float('inf')
        for key, config in full_dataset.object_config.items():
            if abs(config['id'] - raw_id) < min_diff:
                min_diff = abs(config['id'] - raw_id)
                curr_obj_name = config['label']
        
        # --- A. Prepare Ground Truth and Inputs ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() 
        curr_context = full_dataset.C[traj_idx].view(1, 1, -1).to(device)

        # TEMP requires x_full (the time queries for the MLP) 
        x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
        
        # Forward trajectory is required by the forward() method, even if we are forcing p=2
        y1_seq = full_dataset.Y1[traj_idx].unsqueeze(0).to(device)

        # --- B. Prepare Perturbed Sequences ---
        y2_seq_orig = full_dataset.Y2[traj_idx].unsqueeze(0).to(device)
        y2_seq_minus = y2_seq_orig.clone()
        y2_seq_plus = y2_seq_orig.clone()

        condition_sets = {
            "Original": {"seq": y2_seq_orig, "points": []},
            "Shifted -5%": {"seq": y2_seq_minus, "points": []},
            "Shifted +5%": {"seq": y2_seq_plus, "points": []}
        }

        # Apply shifts directly to the condition tensors
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            
            val_orig_norm = y2_seq_orig[0, t_idx, :].clone()
            val_minus_norm = val_orig_norm - 0.05
            val_plus_norm = val_orig_norm + 0.05
            
            y2_seq_minus[0, t_idx, :] = val_minus_norm
            y2_seq_plus[0, t_idx, :] = val_plus_norm
            
            # Denormalize strictly for plotting the condition X's
            val_orig_raw = denormalize_data(val_orig_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            val_minus_raw = denormalize_data(val_minus_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            val_plus_raw = denormalize_data(val_plus_norm, Y_min_vals, Y_max_vals).cpu().numpy()

            condition_sets["Original"]["points"].append({"t": t_val, "raw": val_orig_raw})
            condition_sets["Shifted -5%"]["points"].append({"t": t_val, "raw": val_minus_raw})
            condition_sets["Shifted +5%"]["points"].append({"t": t_val, "raw": val_plus_raw})

        # --- C. Run Inference ---
        predictions = {}
        for cond_name, cond_data in condition_sets.items():
            with torch.no_grad():
                # We do Reconstruction (Inverse -> Inverse) to test conditioning. 
                # p=2 forces the network to route L_I (the inverse latent vector)
                output, _, _, _ = model(
                    y1_seq, 
                    cond_data["seq"], # Pass the perturbed sequence here
                    curr_context, 
                    x_full, 
                    extra_pass=False, 
                    p=2,              # Force Inverse Routing
                    mask_indices_2=eval_mask
                )
            
            # Extract the predicted mean (assuming output is pred_mean)
            _, _, pred_mean_i_norm, _ = output.chunk(4, dim=-1)
            pred_seq_norm = pred_mean_i_norm.squeeze(0)
            
            # Denormalize
            pred_seq_raw = denormalize_data(pred_seq_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            predictions[cond_name] = pred_seq_raw

        # --- D. Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        colors = {"Original": "blue", "Shifted -5%": "orange", "Shifted +5%": "green"}
        
        for col_idx in range(d_y2):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # 2. Plot all Predictions with varying widths so they don't hide each other
            line_styles = {
                "Original": {"width": 4, "alpha": 0.4},
                "Shifted -5%": {"width": 2, "alpha": 0.8},
                "Shifted +5%": {"width": 1, "alpha": 1.0}
            }
            
            for cond_name, pred_data in predictions.items():
                c_color = colors[cond_name]
                w = line_styles[cond_name]["width"]
                a = line_styles[cond_name]["alpha"]
                
                # Plot Predicted Line
                ax.plot(time_steps, pred_data[:, col_idx], 
                        color=c_color, linestyle='--', linewidth=w, alpha=a, label=f'Pred ({cond_name})')
                
                # Plot the Condition Points
                for cond_pt in condition_sets[cond_name]["points"]:
                    ax.scatter(cond_pt["t"], cond_pt["raw"][col_idx], 
                               s=80, marker='X' if cond_name != "Original" else 'o', 
                               color=c_color, label=f'Point ({cond_name})' if cond_pt["t"] == 0 else "")

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Conditioning Perturbation Test (TEMP)\nModel ID: {run_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_multi_object_{num_to_plot}_perturbation_test.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")


if __name__ == "__main__":
    seed = 42
    utils.seed_everything(seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_data_folder = "data/paired_trajectories_insert_place"

    condition_perturbation_test(base_data_folder, num_samples=5, device=device)
