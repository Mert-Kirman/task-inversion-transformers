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
run_id = "run_20260327_180150"
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

def plot_grad_norms():
    """Plots the gradient norms over epochs if they exist."""
    try:
        grad_norms = np.load(os.path.join(save_path, 'grad_norms.npy'))
        plt.figure(figsize=(8, 5))
        plt.plot(grad_norms, label='Gradient Norm')
        window_size = 50
        moving_avg = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(grad_norms)), moving_avg, label=f'{window_size}-Epoch Moving Average')
        plt.title('Gradient Norms Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('L2 Norm of Gradients')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_save = os.path.join(save_path, 'gradient_norms.png')
        plt.savefig(plot_save)
        print(f"Gradient norms plot saved to {plot_save}")
        plt.close()
    except FileNotFoundError:
        print("Gradient norms log not found, skipping gradient norm plot.")

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
        # 1. Grab the full Forward Trajectory
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        
        # 2. Grab the Inverse Trajectory
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        
        # 3. Context
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)

        # Condition points
        eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
        eval_mask[0, 0] = False # Set t=0 to False (Observed)(Single condition point, which is t=0 of the inverse trajectory)
        
        # 4. Run Inference (p=2 forces Decoder to use L_I to generate Forward and Inverse trajectories)
        with torch.no_grad():
            output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
            
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

def evaluate_random_trajectories(base_data_folder, num_samples=6, device='cpu'):
    print(f"\n--- EVALUATING RANDOM TRAJECTORIES ({num_samples} samples) ---")

    # 1. Load Data & Stats
    full_dataset = ReassembleDataset(data_dir=base_data_folder)
    
    checkpoint = torch.load(os.path.join(save_path, "best_model.pth"))
    norm_stats = checkpoint['norm_stats']
    Y_min_vals, Y_max_vals = norm_stats['Y_min'], norm_stats['Y_max']
    C_min_val, C_max_val = norm_stats['C_min'], norm_stats['C_max']

    # Keep a raw copy of Y2 for plotting ground truth before normalization alters it
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
    num_demos = full_dataset.d_N

    # 2. Load Model
    model = temp_model.TempModel(d_x, d_y1, d_y2, d_param).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # 3. Select Random Indices
    num_to_plot = min(num_samples, num_demos)
    indices = random.sample(range(num_demos), num_to_plot)
    
    # Target time steps (Full Sequence)
    time_steps = np.linspace(0, 1, time_len)
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)

    print(f"Evaluating indices: {indices}")

    # Define the evaluation modes (p=1 forces L_F, p=2 forces L_I)
    modes = [
        {'name': 'forward_condition', 'p': 1, 'title': 'Zero-Shot Inversion (Conditioned on Forward Trajectory)'},
        {'name': 'inverse_condition', 'p': 2, 'title': 'Reconstruction (Conditioned on Inverse Trajectory)'}
    ]

    for mode in modes:
        fig, axes = plt.subplots(num_to_plot, d_y2, figsize=(15, 4 * num_to_plot))
        if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

        for row_idx, traj_idx in enumerate(indices):
            # Identify Object Type
            is_paired = full_dataset.valid_inverses[traj_idx]
            curr_obj_name = full_dataset.object_config['round_peg_4']['label'] if is_paired else full_dataset.object_config['square_peg_4']['label']
            
            # --- A. Prepare Ground Truth ---
            curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Place Action (Inverse)
            
            # --- B. Prepare Sequences ---
            y1_seq = full_dataset.Y1[traj_idx].unsqueeze(0).to(device)
            y2_seq = full_dataset.Y2[traj_idx].unsqueeze(0).to(device)
            curr_context = full_dataset.C[traj_idx].view(1, 1, -1).to(device)

            # --- C. Run Inference ---
            with torch.no_grad():
                if mode['p'] == 1:
                    # Condition point(s)
                    condition_points = [60] # t=0.3 of the forward trajectory corresponds to index 60 (since time_len=200)
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
                    eval_mask[0, condition_points] = False # Set condition points to False (Observed)

                    # The p=mode['p'] parameter forces the decoder to use the correct latent vector.
                    output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_1=eval_mask)
                else:
                    # Condition point(s)
                    condition_points = [0] # t=0 of the inverse trajectory corresponds to index 0 (since time_len=200)
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
                    eval_mask[0, condition_points] = False # Set condition points to False (Observed)

                    # The p=mode['p'] parameter forces the decoder to use the correct latent vector.
                    output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_2=eval_mask)

                # Extract the Inverse Trajectory predictions (Mean and Log-Variance)
                _, _, pred_mean_i_norm, pred_std_i_norm = output.chunk(4, dim=-1)
                
                # Convert log variance parameter to standard deviation
                pred_std_i_norm = torch.log(1 + torch.exp(pred_std_i_norm))
                
            # --- D. Denormalize Output ---
            pred_mean_i_norm = pred_mean_i_norm.squeeze(0)
            pred_std_i_norm = pred_std_i_norm.squeeze(0)

            # Denormalize means back to workspace coordinates
            means_pred = denormalize_data(pred_mean_i_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            
            # Standard deviation scales linearly with the range of the workspace
            y_range = (Y_max_vals - Y_min_vals).cpu().numpy()
            stds_pred = pred_std_i_norm.cpu().numpy() * y_range

            # --- E. Plotting ---
            dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
            
            for col_idx in range(d_y2):
                ax = axes[row_idx, col_idx]
                
                # 1. Ground Truth
                ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                        color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
                
                # 2. Prediction
                ax.plot(time_steps, means_pred[:, col_idx], 
                        color='blue', linestyle='--', linewidth=2, label='Pred')
                
                # 3. Uncertainty
                sigma = stds_pred[:, col_idx]
                mean_curve = means_pred[:, col_idx]
                ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                                color='blue', alpha=0.1, label='Uncertainty')
                
                # 4. Condition Points (Only for mode['p'] == 2 since that's where we condition on the inverse trajectory)
                if mode['p'] == 2:
                    ax.scatter(time_steps[0], curr_y_truth_raw[0, col_idx], color='red', s=80, marker='o', label='Condition Point')

                # Labels
                if row_idx == 0:
                    ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize='small', loc='best')

        plt.suptitle(f"{mode['title']}\nModel ID: {run_id}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 
        
        save_file = f'{save_path}/eval_multi_object_{num_to_plot}_{mode["name"]}.png'
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
    
    plot_grad_norms()
    plot_training_progress()
    calculate_success_rates_and_plot(base_data_folder, device=device)
    evaluate_random_trajectories(base_data_folder, num_samples=100, device=device)
