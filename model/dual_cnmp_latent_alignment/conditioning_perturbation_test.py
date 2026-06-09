import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

# Adjust path to find model modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import model.dual_cnmp_latent_alignment.dual_cnmp_model as dual_cnmp_model
import model.model_predict as model_predict
import model.utils as utils

# ================= CONFIGURATION =================
run_id = "run_20260408_235825"
save_path = f"model/dual_cnmp_latent_alignment/save/{run_id}"

# POINT TO THE PAIRED DATA FOLDER
data_path = "data/paired_trajectories_insert_place" 

model_name = "best_model.pth"

# Object Configuration (Must match train.py)
object_config = {
    # ==========================================
    # PAIRED CATEGORIES (The Teachers)
    # ==========================================
    
    # Category 1: Radially Symmetric 
    'round_peg_1':  {'id': 0.0, 'paired': True,  'label': 'Round Peg 1'},
    'round_peg_2':  {'id': 1.0, 'paired': True,  'label': 'Round Peg 2'},
    'round_peg_3':  {'id': 2.0, 'paired': True,  'label': 'Round Peg 3'},
    'round_peg_4':  {'id': 3.0, 'paired': True,  'label': 'Round Peg 4'},
    
    # Category 2: Meshing / Rotational
    'small_gear':   {'id': 4.0, 'paired': True,  'label': 'Small Gear'},
    'medium_gear':  {'id': 5.0, 'paired': True,  'label': 'Medium Gear'},
    'large_gear':   {'id': 6.0, 'paired': True,  'label': 'Large Gear'},
    
    # Category 3: Asymmetric Connectors & Fasteners
    'bnc':          {'id': 7.0, 'paired': True,  'label': 'BNC Connector'},
    'bolt_4':       {'id': 8.0, 'paired': True,  'label': 'Bolt 4 / Nut'},
    'd-sub':        {'id': 9.0, 'paired': True,  'label': 'D-SUB Connector'},
    'ethernet':     {'id': 10.0, 'paired': True,  'label': 'Ethernet Connector'},
    'waterproof':   {'id': 11.0, 'paired': True,  'label': 'Waterproof Connector'},

    # ==========================================
    # UNPAIRED CATEGORIES (Zero-Shot Targets)
    # ==========================================
    
    # Zero-Shot Test 1: Corners & Edges (Highly Geometric, No Rotational Symmetry)
    'square_peg_1': {'id': 12.0, 'paired': False, 'label': 'Square Peg 1 (Unpaired)'},
    'square_peg_2': {'id': 13.0, 'paired': False, 'label': 'Square Peg 2 (Unpaired)'},
    'square_peg_3': {'id': 14.0, 'paired': False, 'label': 'Square Peg 3 (Unpaired)'},
    'square_peg_4': {'id': 15.0, 'paired': False, 'label': 'Square Peg 4 (Unpaired)'},
    
    # Zero-Shot Test 2: Highly Asymmetric Alien Shape
    'usb':          {'id': 16.0, 'paired': False, 'label': 'USB Connector (Unpaired)'}
}
# =================================================

def load_normalization_stats():
    """Loads min/max values used for normalization during training."""
    stats_path = os.path.join(save_path, 'normalization_stats.npy')
    if not os.path.exists(stats_path):
        print(f"Error: Normalization stats not found at {stats_path}")
        sys.exit(1)
    
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # Extract Y stats
    if isinstance(stats['Y_min'], list):
        y_min = torch.stack(stats['Y_min'])
        y_max = torch.stack(stats['Y_max'])
    else:
        y_min = stats['Y_min']
        y_max = stats['Y_max']
    
    # Extract Context stats
    c_min = stats.get('C_min', None)
    c_max = stats.get('C_max', None)
    
    return y_min, y_max, c_min, c_max

def normalize_data(tensor, min_val, max_val):
    """Min-Max normalization to [0, 1]."""
    denominator = max_val - min_val
    denominator[denominator == 0] = 1.0
    return (tensor - min_val) / denominator

def denormalize_data(tensor, min_val, max_val):
    """Reverts [0, 1] data back to original scale."""
    denominator = max_val - min_val
    return tensor * denominator + min_val

def load_matched_data():
    """
    Loads matched insert/place data for ALL configured objects.
    Reconstructs the Context [Avg_X, Avg_Y, Object_ID].
    Returns RAW (un-normalized) tensors and a list of object names for plotting.
    """
    print(f"Loading paired data from {data_path}...")
    
    all_Y1 = []
    all_Y2 = []
    all_C = []
    all_obj_names = [] # To track which trajectory belongs to which object

    for obj_name, config in object_config.items():
        obj_id = config['id']
        obj_dir = os.path.join(data_path, obj_name)
        
        insert_file = os.path.join(obj_dir, 'insert_all.npy')
        place_file = os.path.join(obj_dir, 'place_all.npy')
        
        if not os.path.exists(insert_file) or not os.path.exists(place_file):
            print(f"Warning: Data files not found for {obj_name} in {obj_dir}. Skipping.")
            continue

        # Load arrays of dicts
        insert_data = np.load(insert_file, allow_pickle=True)
        place_data = np.load(place_file, allow_pickle=True)

        # Extract Trajectories (Batch, Time, Dim)
        curr_Y1 = [d['pose'][0][:, :3] for d in insert_data] 
        curr_Y2 = [d['pose'][0][:, :3] for d in place_data]

        # Limit for evaluation (e.g. top 50 per object)
        top_x = min(50, len(curr_Y1))
        curr_Y1 = curr_Y1[:top_x]
        curr_Y2 = curr_Y2[:top_x]
        
        if len(curr_Y1) == 0: continue

        # Stack
        Y1_np = np.stack(curr_Y1)
        Y2_np = np.stack(curr_Y2)
        
        # --- CONTEXT RECONSTRUCTION ---
        # 1. Geometric: (Insert_End_XY + Place_Start_XY) / 2
        insert_ends_xy = Y1_np[:, -1, :2]
        place_starts_xy = Y2_np[:, 0, :2]
        geom_context = (insert_ends_xy + place_starts_xy) / 2.0
        
        # 2. ID Context: Repeated scalar
        id_context = np.full((len(Y1_np), 1), obj_id)
        
        # 3. Combine
        C_np = np.concatenate([geom_context, id_context], axis=1)

        all_Y1.append(Y1_np)
        all_Y2.append(Y2_np)
        all_C.append(C_np)
        all_obj_names.extend([obj_name] * len(Y1_np))
        
        print(f"  Loaded {len(Y1_np)} pairs for {obj_name}")

    # Aggregate
    Y1_raw = torch.tensor(np.concatenate(all_Y1, axis=0), dtype=torch.float32)
    Y2_raw = torch.tensor(np.concatenate(all_Y2, axis=0), dtype=torch.float32)
    C_raw = torch.tensor(np.concatenate(all_C, axis=0), dtype=torch.float32)

    print(f"Total loaded: {Y1_raw.shape[0]} pairs.")
    return Y1_raw, Y2_raw, C_raw, all_obj_names

def condition_perturbation_test(num_samples=6, device='cpu'):
    # 1. Load Norm Stats
    y_min, y_max, c_min, c_max = load_normalization_stats()
    # Move to correct device
    y_min = y_min.to(device)
    y_max = y_max.to(device)
    if c_min is not None:
        c_min = c_min.to(device)
    if c_max is not None:
        c_max = c_max.to(device)
    
    # 2. Load Raw Matched Data
    Y1_raw, Y2_raw, C_raw, obj_names = load_matched_data()
    
    d_x = 1
    d_y1 = Y1_raw.shape[2] 
    d_y2 = Y2_raw.shape[2] 
    d_param = C_raw.shape[1] # Should be 3 (AvgX, AvgY, ID)
    time_len = Y1_raw.shape[1] 
    num_demos = Y1_raw.shape[0]

    # Move data to device
    Y1_raw = Y1_raw.to(device)
    Y2_raw = Y2_raw.to(device)
    C_raw = C_raw.to(device)

    # --- NORMALIZE CONTEXT ---
    # We must normalize C using the stats from training
    C_normalized = C_raw.clone()
    if c_min is not None and c_max is not None:
        C_normalized = normalize_data(C_raw, c_min, c_max)

    # 3. Load Model
    model = dual_cnmp_model.DualCNMP(d_x, d_y1, d_y2, d_param).to(device)
    model_path = os.path.join(save_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model state from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Select Random Indices from the Test Set
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy')).tolist()
    
    num_to_plot = min(num_samples, len(test_idx))
    indices = random.sample(test_idx, num_to_plot)
    
    # 5. Define Condition Points
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [0, -1] # Conditioning at t=0 (Start) and t=1 (End)
    
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        curr_obj_name = obj_names[traj_idx]
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].cpu().numpy() # Place Action (Inverse)
        
        # --- B. Prepare Input (Conditioning on Inverse action) ---
        # We store dictionaries containing the raw value for plotting, and the norm value for the model
        condition_sets = {"Original": [], "Shifted -5%": [], "Shifted +5%": []}
        
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            y_val_raw = Y2_raw[traj_idx, t_idx].unsqueeze(0) 
            y_val_norm_orig = normalize_data(y_val_raw, y_min, y_max)
            
            # Create shifted versions (e.g., +/- 5% of the normalized workspace)
            y_norm_minus = y_val_norm_orig - 0.05
            y_norm_plus = y_val_norm_orig + 0.05
            
            # Denormalize the shifted versions so we can actually plot them correctly
            y_raw_minus = denormalize_data(y_norm_minus, y_min, y_max)
            y_raw_plus = denormalize_data(y_norm_plus, y_min, y_max)

            condition_sets["Original"].append({"t": t_val, "norm": y_val_norm_orig, "raw": y_val_raw})
            condition_sets["Shifted -5%"].append({"t": t_val, "norm": y_norm_minus, "raw": y_raw_minus})
            condition_sets["Shifted +5%"].append({"t": t_val, "norm": y_norm_plus, "raw": y_raw_plus})

        curr_context = C_normalized[traj_idx]

        # --- C. Run Inference (Inverse Mode) ---
        predictions = {}
        for cond_name, cond_list in condition_sets.items():
            # Extract just the [time, normalized_value] pair the model expects
            model_cond = [[c["t"], c["norm"]] for c in cond_list]
            
            with torch.no_grad():
                means_norm, stds_norm = model_predict.predict_inverse_inverse(
                    model, time_len, curr_context, model_cond, d_x, d_y1, d_y2, device=device
                )
                
            # Denormalize immediately
            means_pred = denormalize_data(means_norm, y_min, y_max)
            stds_pred = stds_norm * (y_max - y_min)
            predictions[cond_name] = {"mean": means_pred, "std": stds_pred}

        # --- E. Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        colors = {"Original": "blue", "Shifted -5%": "orange", "Shifted +5%": "green"}
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # 2. Plot all Predictions and their corresponding Condition Points
            for cond_name, pred_data in predictions.items():
                c_color = colors[cond_name]
                
                # Plot Predicted Line
                ax.plot(time_steps, pred_data["mean"][:, col_idx].cpu().numpy(), 
                        color=c_color, linestyle='--', linewidth=2, label=f'Pred ({cond_name})')
                
                # Plot the Condition Points using the RAW coordinate
                for cond_pt in condition_sets[cond_name]:
                    ax.scatter(cond_pt["t"], cond_pt["raw"][:, col_idx].cpu().numpy(), 
                               s=80, marker='X' if cond_name != "Original" else 'o', 
                               color=c_color, label=f'Point ({cond_name})' if cond_pt["t"] == 0 else "")

            # 3. Uncertainty (Only plot for the original to avoid clutter)
            sigma = predictions["Original"]["std"][:, col_idx].cpu().numpy()
            mean_curve = predictions["Original"]["mean"][:, col_idx].cpu().numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty (Original)')

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Conditioning Perturbation Test\nModel: {model_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_multi_object_{num_to_plot}_perturbation_test.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")


if __name__ == "__main__":
    utils.seed_everything(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    condition_perturbation_test(num_samples=5, device=device)