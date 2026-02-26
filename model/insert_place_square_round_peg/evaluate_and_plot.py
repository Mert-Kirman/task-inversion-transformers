import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Adjust path to find model modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import model.insert_place_square_round_peg.dual_enc_dec_model as dual_enc_dec_cnmp
import model.model_predict as model_predict
import model.utils as utils

# ================= CONFIGURATION =================
run_id = "run_1767992349.943897"
save_path = f"model/insert_place_square_round_peg/save/{run_id}"

# POINT TO THE PAIRED DATA FOLDER
data_path = "data/paired_trajectories_insert_place" 

model_name = "best_model.pth"

# Object Configuration (Must match train.py)
object_config = {
    'round_peg_4':  {'id': 0.0, 'label': 'Round Peg 4 (Source)'},
    'square_peg_4': {'id': 1.0, 'label': 'Square Peg 4 (Target)'} 
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

def plot_training_progress():
    """Plots loss and error curves if they exist."""
    try:
        train_err = np.load(f'{save_path}/training_errors_mse.npy')
        val_err = np.load(f'{save_path}/validation_errors_mse.npy')
        losses = np.load(f'{save_path}/losses_log_prob.npy')

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(losses, label='Training Loss', color='orange', alpha=0.7)
        plt.title('Log Probability Loss')
        plt.xlabel('Step')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_err, label='Train MSE', color='blue')
        plt.plot(val_err, label='Val MSE', color='red', linestyle='--')
        plt.title('Reconstruction Error (MSE)')
        plt.xlabel('Epoch (x1000)')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(train_err, label='Train MSE', color='blue')
        plt.plot(val_err, label='Val MSE', color='red', linestyle='--')
        plt.title('MSE (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plot_save = f'{save_path}/training_progress_multi.png'
        plt.savefig(plot_save)
        print(f"Training progress saved to {plot_save}")
        plt.close()
    except FileNotFoundError:
        print("Training logs not found, skipping progress plot.")

def calculate_success_rates_and_plot(device='cpu'):
    """
    Evaluates success based on Start (t=0) and End (t=1) point accuracy.
    Threshold: 5% (Strict) and 10% (Relaxed) of the global data range (per dimension).
    """
    print("\n--- CALCULATING SUCCESS RATES & PLOTTING ---")
    
    # Load Data & Stats
    y_min, y_max, c_min, c_max = load_normalization_stats()
    Y1_raw, Y2_raw, C_raw, obj_names = load_matched_data()
    
    # Determine Thresholds
    global_range = (y_max - y_min).numpy()
    
    # Define scenarios: Label, Percentage, Threshold Vector
    scenarios = [
        {'label': '5% (Strict)', 'pct': 0.05, 'thresh': 0.05 * global_range},
        {'label': '10% (Relaxed)', 'pct': 0.10, 'thresh': 0.10 * global_range}
    ]
    
    print(f"Global Range (X, Y, Z): {global_range}")
    
    # Load Model
    d_x = 1
    d_y1 = Y1_raw.shape[2] 
    d_y2 = Y2_raw.shape[2] 
    d_param = C_raw.shape[1] 
    time_len = Y1_raw.shape[1] 
    
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param).to(device)
    model_path = os.path.join(save_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Normalize Inputs
    C_normalized = C_raw.clone()
    if c_min is not None and c_max is not None:
        C_normalized = normalize_data(C_raw, c_min, c_max)

    # Run Inference ONCE for all data
    t_steps = np.linspace(0, 1, time_len)
    cond_idx = 0 # t=0
    
    # Store predictions to avoid re-running model for each threshold
    predictions = [] # List of (pred_traj, gt_traj, obj_name)
    
    print("Running Inference...")
    for i in range(len(Y2_raw)):
        # Prepare Condition
        y_cond_raw = Y2_raw[i, cond_idx:cond_idx+1]
        y_cond_norm = normalize_data(y_cond_raw, y_min, y_max)
        cond_pts = [[t_steps[cond_idx], y_cond_norm]]
        
        curr_context = C_normalized[i]
        
        with torch.no_grad():
             means_norm, _ = model_predict.predict_inverse_inverse(
                model, time_len, curr_context, cond_pts, d_x, d_y1, d_y2, device=device
            )
        
        pred_traj = denormalize_data(means_norm, y_min, y_max).numpy()
        gt_traj = Y2_raw[i].numpy()
        
        predictions.append({
            'pred': pred_traj,
            'gt': gt_traj,
            'obj': obj_names[i]
        })

    # Evaluate Success for Each Scenario
    # Structure: results[scenario_label][obj_name] = success_rate
    final_stats = {s['label']: {} for s in scenarios}
    obj_counts = {}

    for s in scenarios:
        print(f"\nEvaluating Scenario: {s['label']}")
        thresholds = s['thresh']
        
        # Temp counters
        counts = {} 
        
        for p in predictions:
            obj = p['obj']
            if obj not in counts: counts[obj] = {'total': 0, 'success': 0}
            if obj not in obj_counts: obj_counts[obj] = 0 # Track total counts once
            
            counts[obj]['total'] += 1
            if s == scenarios[0]: obj_counts[obj] += 1
            
            # Check Start (t=0) and End (t=1) - X(0) and Y(1) ONLY
            pred_start = p['pred'][0]
            gt_start = p['gt'][0]
            start_ok = np.all(np.abs(pred_start - gt_start)[:2] <= thresholds[:2])
            
            pred_end = p['pred'][-1]
            gt_end = p['gt'][-1]
            end_ok = np.all(np.abs(pred_end - gt_end)[:2] <= thresholds[:2])
            
            if start_ok and end_ok:
                counts[obj]['success'] += 1
        
        # Calculate Rates
        for obj, stats in counts.items():
            rate = (stats['success'] / stats['total']) * 100
            final_stats[s['label']][obj] = rate
            print(f"  {obj}: {rate:.2f}% ({stats['success']}/{stats['total']})")

    # Generate Bar Chart
    print("\nGenerating Bar Chart...")
    
    # Data Preparation
    labels = [object_config[o]['label'] for o in object_config] # ["Round Peg (Source)", "Square Peg (Target)"]
    obj_keys = list(object_config.keys()) # ['round_peg_4', 'square_peg_4']
    
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each scenario
    rects1 = ax.bar(x - width/2, [final_stats['5% (Strict)'][k] for k in obj_keys], width, label='5% Tolerance (Strict)', color='#d9534f')
    rects2 = ax.bar(x + width/2, [final_stats['10% (Relaxed)'][k] for k in obj_keys], width, label='10% Tolerance (Relaxed)', color='#5bc0de')
    
    # Styling
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Task Extrapolation Success Rates\n(Round vs. Square Peg)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add Value Labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'success_rate_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Bar chart saved to {plot_path}")

def evaluate_random_trajectories(num_samples=6, device='cpu'):
    # 1. Load Norm Stats
    y_min, y_max, c_min, c_max = load_normalization_stats()
    
    # 2. Load Raw Matched Data
    Y1_raw, Y2_raw, C_raw, obj_names = load_matched_data()
    
    d_x = 1
    d_y1 = Y1_raw.shape[2] 
    d_y2 = Y2_raw.shape[2] 
    d_param = C_raw.shape[1] # Should be 3 (AvgX, AvgY, ID)
    time_len = Y1_raw.shape[1] 
    num_demos = Y1_raw.shape[0]

    # --- NORMALIZE CONTEXT ---
    # We must normalize C using the stats from training
    C_normalized = C_raw.clone()
    if c_min is not None and c_max is not None:
        C_normalized = normalize_data(C_raw, c_min, c_max)

    # 3. Load Model
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param).to(device)
    model_path = os.path.join(save_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model state from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Select Random Indices
    num_to_plot = min(num_samples, num_demos)
    indices = random.sample(range(num_demos), num_to_plot)
    
    # 5. Define Condition Points
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [300] # Conditioning at t=0.3 (During Insert)
    
    # 6. Plot Setup
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        curr_obj_name = obj_names[traj_idx]
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Place Action (Inverse)
        
        # --- B. Prepare Input (Conditioning on Insert action) ---
        # The model is predicting Y2 (Place) given Y1 (Insert)
        condition_points = []
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            y_val_raw = Y1_raw[traj_idx, t_idx:t_idx+1]
            y_val_norm = normalize_data(y_val_raw, y_min, y_max)
            condition_points.append([t_val, y_val_norm])
        
        curr_context = C_normalized[traj_idx]

        # --- C. Run Inference (Inverse Mode) ---
        with torch.no_grad():
            means_norm, stds_norm = model_predict.predict_inverse(
                model, time_len, curr_context, condition_points, d_x, d_y1, d_y2, device=device
            )
            
        # --- D. Denormalize Output ---
        means_pred = denormalize_data(means_norm, y_min, y_max)
        stds_pred = stds_norm * (y_max - y_min)

        # --- E. Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # 2. Prediction
            ax.plot(time_steps, means_pred[:, col_idx].numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Pred')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].numpy()
            mean_curve = means_pred[:, col_idx].numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty')

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                # Add Object Name to Y-Label for Clarity
                ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Inverse Task Prediction (Round Peg vs Square Peg)\nModel: {model_name} | ID Context Included", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_multi_object_{num_to_plot}_forward_condition.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")

    # Condition from place action at t=0
    cond_step_indices = [0] # Conditioning at t=0.3 (During Insert)
    
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        curr_obj_name = obj_names[traj_idx]
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Place Action (Inverse)
        
        # --- B. Prepare Input (Conditioning on Insert action) ---
        # The model is predicting Y2 (Place) given Y1 (Insert)
        condition_points = []
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            y_val_raw = Y2_raw[traj_idx, t_idx:t_idx+1]
            y_val_norm = normalize_data(y_val_raw, y_min, y_max)
            condition_points.append([t_val, y_val_norm])
        
        curr_context = C_normalized[traj_idx]

        # --- C. Run Inference (Inverse Mode) ---
        with torch.no_grad():
            means_norm, stds_norm = model_predict.predict_inverse_inverse(
                model, time_len, curr_context, condition_points, d_x, d_y1, d_y2, device=device
            )
            
        # --- D. Denormalize Output ---
        means_pred = denormalize_data(means_norm, y_min, y_max)
        stds_pred = stds_norm * (y_max - y_min)

        # --- E. Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # 2. Prediction
            ax.plot(time_steps, means_pred[:, col_idx].numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Pred')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].numpy()
            mean_curve = means_pred[:, col_idx].numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty')
            
            # 4. Conditioning Point
            cond_y_raw = Y2_raw[traj_idx, cond_step_indices[0], col_idx]
            ax.scatter(time_steps[cond_step_indices[0]], cond_y_raw, 
                       color='red', s=80, marker='o', label='Condition Point')

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                # Add Object Name to Y-Label for Clarity
                ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Inverse Task Prediction (Round Peg vs Square Peg)\nModel: {model_name} | ID Context Included", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_multi_object_{num_to_plot}_inverse_condition.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")

if __name__ == "__main__":
    utils.seed_everything(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    plot_training_progress()
    calculate_success_rates_and_plot(device=device)
    evaluate_random_trajectories(num_samples=100, device=device)
