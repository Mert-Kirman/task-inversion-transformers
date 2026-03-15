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

from dataset import ReassembleDataset
import model.transformer_encoded_movement_primitive.temp_model as temp_model
import model.model_predict as model_predict
import model.utils as utils

# ================= CONFIGURATION =================
run_id = "run_20260315_222036"
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

def plot_training_progress():
    """Plots loss and error curves if they exist."""
    try:
        composite_losses = np.load(f'{save_path}/composite_losses.npy')
        
        train_fwd_mse = np.load(f'{save_path}/train_fwd_mse.npy')
        train_inv_mse = np.load(f'{save_path}/train_inv_mse.npy')

        val_fwd_mse = np.load(f'{save_path}/val_fwd_mse.npy')
        val_inv_mse = np.load(f'{save_path}/val_inv_mse.npy')

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(composite_losses, label='Composite Loss (Log Prob + Latent Alignment MSE)')
        # Add a moving average for cleaner visualization
        window_size = 50
        moving_avg = np.convolve(composite_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(composite_losses)), moving_avg, label=f'{window_size}-Epoch Moving Average')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_fwd_mse, label='Train Forward MSE')
        plt.plot(val_fwd_mse, label='Val Forward MSE')
        plt.title('Forward MSE')
        plt.xlabel('Epoch (x50)')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_inv_mse, label='Train Inverse MSE')
        plt.plot(val_inv_mse, label='Val Inverse MSE')
        plt.title('Inverse MSE')
        plt.xlabel('Epoch (x50)')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plot_save = f'{save_path}/training_progress.png'
        plt.savefig(plot_save)
        print(f"Training progress saved to {plot_save}")
        plt.close()
    except FileNotFoundError:
        print("Training logs not found, skipping progress plot.")

def calculate_success_rates_and_plot(base_data_folder, device='cpu'):
    """
    Evaluates success based on Start (t=0) and End (t=1) point accuracy.
    Threshold: 5% (Strict) and 10% (Relaxed) of the global data range (per dimension).
    """
    print("\n--- CALCULATING SUCCESS RATES & PLOTTING ---")

    # Load Data
    full_dataset = ReassembleDataset(data_dir=base_data_folder)

    # Load Normalization Stats
    checkpoint = torch.load(os.path.join(save_path, "best_model.pth"))
    norm_stats = checkpoint['norm_stats']
    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']

    # Normalize data
    full_dataset.Y1, full_dataset.Y2, full_dataset.C = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val)
    
    # Determine Thresholds
    global_range = Y_max_vals - Y_min_vals
    
    # Define scenarios: Label, Percentage, Threshold Vector
    scenarios = [
        {'label': '5% (Strict)', 'pct': 0.05, 'thresh': 0.05 * global_range},
        {'label': '10% (Relaxed)', 'pct': 0.10, 'thresh': 0.10 * global_range}
    ]
    
    print(f"Global Range (X, Y, Z): {global_range}")
    
    # Data dimensions
    d_x = full_dataset.d_x
    d_y1 = full_dataset.d_y1
    d_y2 = full_dataset.d_y2
    d_param = full_dataset.d_param
    time_len = full_dataset.time_len

    # Load Model
    model = temp_model.TempModel(d_x, d_y1, d_y2, d_param).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Move bounds to device for denormalization
    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # Prepare the target time steps (all 200 points)
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
    
    predictions = []
    
    print("Running Zero-Shot Inference (Conditioning on Full Forward Trajectory)...")
    for i in range(len(full_dataset)):
        # 1. Grab the full Forward Trajectory (Condition)
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        
        # 2. Grab the Inverse Trajectory (Passed for signature, but ignored by p=1)
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        
        # 3. Context
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)
        
        # 4. Run Inference (p=1 forces Decoder to use L_F to generate Inverse)
        with torch.no_grad():
            output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=1)
            
        # 5. Extract the Predicted Inverse Trajectory
        _, _, pred_mean_i, _ = output.chunk(4, dim=-1)
        
        # 6. Denormalize Predictions and Ground Truth to physical scale
        pred_traj = denormalize_data(pred_mean_i.squeeze(0), Y_min_vals, Y_max_vals).cpu().numpy()
        gt_traj = denormalize_data(full_dataset.Y2[i].to(device), Y_min_vals, Y_max_vals).cpu().numpy()
        
        # 7. Identify Object Type (Round = Paired/True, Square = Unpaired/False)
        is_paired = full_dataset.valid_inverses[i]
        obj_key = 'round_peg_4' if is_paired else 'square_peg_4'
        
        predictions.append({
            'pred': pred_traj,
            'gt': gt_traj,
            'obj': obj_key
        })

    # Evaluate Success for Each Scenario
    # Structure: results[scenario_label][obj_name] = success_rate
    final_stats = {s['label']: {} for s in scenarios}
    obj_counts = {}

    for s in scenarios:
        print(f"\nEvaluating Scenario: {s['label']}")
        thresholds = s['thresh'].cpu().numpy()
        
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
    labels = [full_dataset.object_config[o]['label'] for o in full_dataset.object_config] # ["Round Peg (Source)", "Square Peg (Target)"]
    obj_keys = list(full_dataset.object_config.keys()) # ['round_peg_4', 'square_peg_4']
    
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

    # 4. Select Random Indices
    num_to_plot = min(num_samples, num_demos)
    indices = random.sample(range(num_demos), num_to_plot)
    
    # 5. Define Condition Points
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [60] # Conditioning at t=0.3 (During Insert)
    
    # 6. Plot Setup
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        curr_obj_name = obj_names[traj_idx]
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].cpu().numpy() # Place Action (Inverse)
        
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
            ax.plot(time_steps, means_pred[:, col_idx].cpu().numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Pred')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].cpu().numpy()
            mean_curve = means_pred[:, col_idx].cpu().numpy()
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
    cond_step_indices = [0]
    
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        curr_obj_name = obj_names[traj_idx]
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].cpu().numpy() # Place Action (Inverse)
        
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
            ax.plot(time_steps, means_pred[:, col_idx].cpu().numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Pred')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].cpu().numpy()
            mean_curve = means_pred[:, col_idx].cpu().numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty')
            
            # 4. Conditioning Point
            cond_y_raw = Y2_raw[traj_idx, cond_step_indices[0], col_idx].cpu().numpy()
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
    seed = 42
    utils.seed_everything(seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_data_folder = "data/paired_trajectories_insert_place"
    
    plot_training_progress()
    calculate_success_rates_and_plot(base_data_folder, device=device)
    evaluate_random_trajectories(num_samples=100, device=device)
