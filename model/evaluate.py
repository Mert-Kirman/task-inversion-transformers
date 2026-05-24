import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Adjust path to find model modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import ReassembleDataset
import model.transformer_encoded_diffusion_policy.tedp_model as tedp_model
import model.model_predict as model_predict
from model.utils import seed_everything

from scipy.signal import savgol_filter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNMP/TEMP/TEDP models on Reassemble/Synthetic datasets.")
    parser.add_argument("--model", type=str, required=True, choices=["cnmp", "temp_vanilla", "temp_unmasked_pooling", "tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"], help="Which model architecture to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, choices=["reassemble", "synthetic_small", "synthetic_large"], help="Which dataset to evaluate on.")
    parser.add_argument("--run_id", type=str, required=True, help="Identifier for the model run to load and evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args

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

def plot_grad_norms(save_path):
    """Plots the gradient norms over epochs if they exist."""
    try:
        grad_norms = np.load(os.path.join(save_path, 'grad_norms.npy'))
        plt.figure(figsize=(8, 5))
        plt.plot(grad_norms, label='Gradient Norm')
        window_size = 20
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

def plot_training_progress_cnmp(save_path):
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

def plot_training_progress_temp_tedp(save_path):
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
        window_size = 20
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
        plt.xlabel('Epoch (x20)')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_inv_mse, label='Train Inverse MSE')
        plt.plot(val_inv_mse, label='Val Inverse MSE')
        plt.title('Inverse MSE')
        plt.xlabel('Epoch (x20)')
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

def calculate_success_rates_and_plot(save_path, full_dataset, norm_stats, model, args, device='cpu'):
    """
    Evaluates success based on Start (t=0) and End (t=1) point accuracy.
    Threshold: 5% (Strict) and 10% (Relaxed) of the global data range (per dimension).
    """
    print("\n--- CALCULATING SUCCESS RATES & PLOTTING ---")

    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']
    
    # Determine Thresholds
    global_range = Y_max_vals - Y_min_vals
    
    # Define scenarios: Label, Percentage, Threshold Vector
    scenarios = [
        {'label': '5% (Strict)', 'pct': 0.05, 'thresh': 0.05 * global_range},
        {'label': '10% (Relaxed)', 'pct': 0.10, 'thresh': 0.10 * global_range}
    ]
    
    print(f"Global Range (X, Y, Z): {global_range}")
    
    time_len = full_dataset.time_len

    # Move bounds to device for denormalization
    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # Load the hidden test indices
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy'))

    # Prepare the target time steps (all 200 points)
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
    predictions = []
    
    print("Running Zero-Shot Inference (Conditioning on start and end points of inverse trajectories)...")
    for i in test_idx:
        # Grab the full Forward Trajectory
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        
        # Grab the Inverse Trajectory
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        
        # Context
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)

        # Condition points
        condition_points = [0, -1] # t=0 and t=1 of the inverse trajectory corresponds to indices 0 and 199 (since time_len=200)
        eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
        eval_mask[0, condition_points] = False # Set condition points to False (Observed)
        
        # Run Inference
        with torch.no_grad():
            if args.model.startswith('cnmp'):
                # Prepare Condition
                cond_pts = []
                for idx in condition_points:
                    cond_pts.append([x_full[0, idx], y2_seq[0, idx]])

                means_norm, _ = model_predict.predict_inverse_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                pred_seq = means_norm
            elif args.model.startswith('temp'):
                output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
                # Extract the Predicted Inverse Trajectory
                _, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                pred_seq = pred_mean_i
            elif args.model.startswith('tedp'):
                pred_seq = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
        
        # Denormalize Predictions and Ground Truth to physical scale
        pred_traj = denormalize_data(pred_seq.squeeze(0), Y_min_vals, Y_max_vals).cpu().numpy()
        gt_traj = denormalize_data(full_dataset.Y2[i].to(device), Y_min_vals, Y_max_vals).cpu().numpy()
        
        # Identify Object Type dynamically via Context ID
        curr_context_norm = full_dataset.C[i]
        raw_id = denormalize_data(curr_context_norm, C_min_val, C_max_val)[-1].item()
        
        obj_key = None
        min_diff = float('inf')
        for key, config in full_dataset.object_config.items():
            diff = abs(config['id'] - raw_id)
            if diff < min_diff:
                min_diff = diff
                obj_key = key
        
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
    
    # Only plot objects that actually appeared in the test set evaluation
    evaluated_obj_keys = list(obj_counts.keys())
    labels = [f"{full_dataset.object_config[k]['label']} (n={obj_counts[k]})" for k in evaluated_obj_keys]
    
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot bars using safe .get() in case a category was missing in one scenario
    rects1 = ax.bar(x - width/2, [final_stats['5% (Strict)'].get(k, 0) for k in evaluated_obj_keys], width, label='5% Tolerance (Strict)', color='#d9534f')
    rects2 = ax.bar(x + width/2, [final_stats['10% (Relaxed)'].get(k, 0) for k in evaluated_obj_keys], width, label='10% Tolerance (Relaxed)', color='#5bc0de')
    
    # Styling
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Task Extrapolation Success Rates (Test Set)\n{args.model.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Rotate labels so they don't overlap
    ax.set_xticklabels(labels, fontsize=8, fontweight='bold', rotation=45, ha='right')
    ax.set_ylim(0, 130)
    ax.legend(loc='upper center', ncol=2, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add Value Labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontweight='bold',
                        fontsize=6,
                        rotation=45)    # Rotate text to prevent horizontal collision

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'success_rate_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Bar chart saved to {plot_path}")

def calculate_continuous_errors_and_plot(save_path, full_dataset, norm_stats, model, args, device='cpu'):
    print("\n--- CALCULATING CONTINUOUS ERRORS (CM) & PLOTTING ---")

    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']
    
    time_len = full_dataset.time_len

    # Move bounds to device for denormalization
    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # Setup Data Structures
    metrics = ['Euclidean (3D)', 'X-Axis (Left/Right)', 'Y-Axis (Forward/Back)', 'Z-Axis (Depth)']
    start_errors = {m: {} for m in metrics}
    end_errors = {m: {} for m in metrics}

    # Load the hidden test indices
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy'))
    print(f"Evaluating continuous errors on {len(test_idx)} test samples...")
    
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)

    # Run Inference Loop
    for i in test_idx:
        # Grab the full Forward Trajectory
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        
        # Grab the Inverse Trajectory
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        
        # Context
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)

        # Condition points
        condition_points = [0, -1] # t=0 and t=1 of the inverse trajectory corresponds to indices 0 and 199 (since time_len=200)
        eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
        eval_mask[0, condition_points] = False # Set condition points to False (Observed)
        
        # Run Inference
        with torch.no_grad():
            if args.model.startswith('cnmp'):
                # Prepare Condition
                cond_pts = []
                for idx in condition_points:
                    cond_pts.append([x_full[0, idx], y2_seq[0, idx]])

                means_norm, _ = model_predict.predict_inverse_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                pred_seq = means_norm
            elif args.model.startswith('temp'):
                output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
                # Extract the Predicted Inverse Trajectory
                _, _, pred_mean_i, _ = output.chunk(4, dim=-1)
                pred_seq = pred_mean_i
            elif args.model.startswith('tedp'):
                pred_seq = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
        
        # Denormalize Predictions and Ground Truth to physical scale
        pred_traj = denormalize_data(pred_seq.squeeze(0), Y_min_vals, Y_max_vals).cpu().numpy()
        gt_traj = denormalize_data(full_dataset.Y2[i].to(device), Y_min_vals, Y_max_vals).cpu().numpy()
        
        # Identify Object Type dynamically via Context ID
        curr_context_norm = full_dataset.C[i]
        raw_id = denormalize_data(curr_context_norm, C_min_val, C_max_val)[-1].item()
        
        obj_key = None
        min_diff = float('inf')
        for key, config in full_dataset.object_config.items():
            diff = abs(config['id'] - raw_id)
            if diff < min_diff:
                min_diff = diff
                obj_key = key
        
        if obj_key not in start_errors['Euclidean (3D)']:
            for m in metrics:
                start_errors[m][obj_key] = []
                end_errors[m][obj_key] = []
                
        # Calculate Differences in cm
        diff_start = (pred_traj[0, :3] - gt_traj[0, :3]) * 100.0
        diff_end = (pred_traj[-1, :3] - gt_traj[-1, :3]) * 100.0
        
        start_errors['Euclidean (3D)'][obj_key].append(np.linalg.norm(diff_start))
        start_errors['X-Axis (Left/Right)'][obj_key].append(abs(diff_start[0]))
        start_errors['Y-Axis (Forward/Back)'][obj_key].append(abs(diff_start[1]))
        start_errors['Z-Axis (Depth)'][obj_key].append(abs(diff_start[2]))
        
        end_errors['Euclidean (3D)'][obj_key].append(np.linalg.norm(diff_end))
        end_errors['X-Axis (Left/Right)'][obj_key].append(abs(diff_end[0]))
        end_errors['Y-Axis (Forward/Back)'][obj_key].append(abs(diff_end[1]))
        end_errors['Z-Axis (Depth)'][obj_key].append(abs(diff_end[2]))

    # Helper function to plot a set of 4 Violins using Seaborn
    def create_violin_figure(error_data, time_title, filename):
        evaluated_obj_keys = list(error_data['Euclidean (3D)'].keys())
        
        # Map object keys to their label with the (n=X) count included
        label_map = {k: f"{full_dataset.object_config[k]['label']} (n={len(error_data['Euclidean (3D)'][k])})" for k in evaluated_obj_keys}
        
        # Convert the dictionary into a Pandas DataFrame for Seaborn
        rows = []
        for m in metrics:
            for k in evaluated_obj_keys:
                obj_label = label_map[k]
                for val in error_data[m][k]:
                    rows.append({'Metric': m, 'Object': obj_label, 'Error (cm)': val})
        df = pd.DataFrame(rows)

        # Save the error data for further plotting
        csv_filename = filename.replace('.png', '.csv')
        df.to_csv(os.path.join(save_path, csv_filename), index=False)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 18))
        fig.suptitle(f'Trajectory {time_title} Deviation', fontsize=18, fontweight='bold', y=0.98)

        for idx, m in enumerate(metrics):
            ax = axes[idx]
            df_metric = df[df['Metric'] == m]
            
            # Seaborn Violin Plot
            sns.violinplot(
                data=df_metric, 
                x='Object', 
                y='Error (cm)', 
                ax=ax,
                color='#5bc0de',
                linewidth=1.5,
                inner='box',
                cut=0,   # Prevent the violin from drawing density below 0 cm
                density_norm='width'
            )
            
            ax.set_title(m, fontsize=14, fontweight='bold')
            ax.set_ylabel('Error (cm)', fontsize=12, fontweight='bold')
            ax.set_xlabel('') # Clear redundant x-axis label
            
            # Only show object names on the very bottom plot
            if idx == 3:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight='bold', rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
                
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97]) 
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Saved: {plot_path}")

    # Helper function to calculate and save numerical statistics
    def save_numerical_statistics(error_data, time_title, filename):
        print(f"\n=== Numerical Statistics: {time_title} ===")
        stats_file_path = os.path.join(save_path, filename)
        
        with open(stats_file_path, 'w') as f:
            for m in metrics:
                header = f"\n--- {m} ---"
                print(header)
                f.write(header + "\n")
                
                evaluated_obj_keys = list(error_data[m].keys())
                for k in evaluated_obj_keys:
                    err_list = error_data[m][k]
                    if len(err_list) == 0:
                        continue
                        
                    mean_val = np.mean(err_list)
                    std_val = np.std(err_list)
                    
                    # Use the raw object name from object_config, plus (n=X)
                    obj_label = f"{full_dataset.object_config[k]['label']} (n={len(err_list)})"
                    line = f"{obj_label:<35} | Mean: {mean_val:>5.2f} cm | Std: {std_val:>5.2f} cm"
                    
                    print(line)
                    f.write(line + "\n")
                    
        print(f"\nStatistics saved to {stats_file_path}")

    # Generate the two separate figure files
    print("\nGenerating Seaborn Violin Plots...")
    create_violin_figure(start_errors, "Start Point (t=0)", 'continuous_error_violins_start.png')
    create_violin_figure(end_errors, "End Point (t=1)", 'continuous_error_violins_end.png')

    # Generate and save the numerical statistics
    save_numerical_statistics(start_errors, "Start Point (t=0)", 'continuous_errors_stats_start.txt')
    save_numerical_statistics(end_errors, "End Point (t=1)", 'continuous_errors_stats_end.txt')

def predict_random_trajectories(save_path, full_dataset, Y2_raw, norm_stats, model, args, num_samples=6, device='cpu'):
    print(f"\n--- PREDICTING RANDOM TRAJECTORIES ({num_samples} samples) ---")

    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']

    time_len = full_dataset.time_len

    # Move bounds to device for denormalization
    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)

    # Load test indices and sample from them
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy'))
    test_idx_list = test_idx.tolist()
    
    num_to_plot = min(num_samples, len(test_idx_list))
    indices = random.sample(test_idx_list, num_to_plot)
    
    # Target time steps (Full Sequence)
    time_steps = np.linspace(0, 1, time_len)
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)

    print(f"Plotting trajectories for test indices: {indices}")

    # Define the evaluation modes (p=1 forces L_F, p=2 forces L_I)
    modes = [
        {'name': 'forward_condition', 'p': 1, 'title': 'Conditioned on Forward Trajectory'},
        {'name': 'inverse_condition', 'p': 2, 'title': 'Conditioned on Inverse Trajectory'}
    ]

    for mode in modes:
        fig, axes = plt.subplots(num_to_plot, full_dataset.d_y2, figsize=(15, 4 * num_to_plot))
        if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

        for row_idx, traj_idx in enumerate(indices):
            # Identify Object Type dynamically via Context ID
            curr_context_norm = full_dataset.C[traj_idx]
            raw_id = denormalize_data(curr_context_norm, C_min_val, C_max_val)[-1].item()
            
            curr_obj_name = "Unknown"
            min_diff = float('inf')
            for key, config in full_dataset.object_config.items():
                diff = abs(config['id'] - raw_id)
                if diff < min_diff:
                    min_diff = diff
                    curr_obj_name = config['label']
            
            # --- Prepare Ground Truth ---
            curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Place Action (Inverse)
            
            # --- Prepare Sequences ---
            y1_seq = full_dataset.Y1[traj_idx].unsqueeze(0).to(device)
            y2_seq = full_dataset.Y2[traj_idx].unsqueeze(0).to(device)
            curr_context = full_dataset.C[traj_idx].view(1, 1, -1).to(device)

            # --- Run Inference ---
            samples = []
            with torch.no_grad():
                if mode['p'] == 1:
                    # Condition on the Forward Trajectory
                    condition_points = [60] # t=0.3 of the forward trajectory corresponds to index 60 (since time_len=200)
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
                    eval_mask[0, condition_points] = False # Set condition points to False (Observed)

                    if args.model.startswith('cnmp'):
                        # Prepare Condition
                        cond_pts = []
                        for idx in condition_points:
                            cond_pts.append([x_full[0, idx], y2_seq[0, idx]])

                        pred_mean_i_norm, pred_std_i_norm = model_predict.predict_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                    elif args.model.startswith('temp'):
                        output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_1=eval_mask) # The p=mode['p'] parameter forces the decoder to use the correct latent vector
                        _, _, pred_mean_i_norm, pred_std_i_norm = output.chunk(4, dim=-1) # Extract the Inverse Trajectory predictions (Mean and Log-Variance)
                        pred_std_i_norm = torch.log(1 + torch.exp(pred_std_i_norm)) # Convert log variance parameter to standard deviation

                        pred_mean_i_norm = pred_mean_i_norm.squeeze(0)
                        pred_std_i_norm = pred_std_i_norm.squeeze(0)
                    elif args.model.startswith('tedp'):
                        num_mc_samples = 6 # Number of diffusion generations for uncertainty estimation
                        for _ in range(num_mc_samples):
                            pred = model.sample(cond_seq=y1_seq, params=curr_context, mask_indices=eval_mask, source_dim='y1', target_dim='y2', time_len=time_len)
                            samples.append(pred.squeeze(0))
                else:
                    # Condition on Inverse Trajectory
                    condition_points = [0, -1] 
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
                    eval_mask[0, condition_points] = False 

                    if args.model.startswith('cnmp'):
                        # Prepare Condition
                        cond_pts = []
                        for idx in condition_points:
                            cond_pts.append([x_full[0, idx], y2_seq[0, idx]])

                        pred_mean_i_norm, pred_std_i_norm = model_predict.predict_inverse_inverse(model, time_len, curr_context, condition_points, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                    elif args.model.startswith('temp'):
                        output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_2=eval_mask)
                        _, _, pred_mean_i_norm, pred_std_i_norm = output.chunk(4, dim=-1) # Extract the Inverse Trajectory predictions (Mean and Log-Variance)
                        pred_std_i_norm = torch.log(1 + torch.exp(pred_std_i_norm)) # Convert log variance parameter to standard deviation

                        pred_mean_i_norm = pred_mean_i_norm.squeeze(0)
                        pred_std_i_norm = pred_std_i_norm.squeeze(0)
                    elif args.model.startswith('tedp'):
                        num_mc_samples = 6
                        for _ in range(num_mc_samples):
                            pred = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
                            samples.append(pred.squeeze(0))

                        # Calculate Empirical Mean and Standard Deviation across the 10 samples
                        samples_tensor = torch.stack(samples) # Shape: (10, 200, 3)
                        pred_mean_i_norm = samples_tensor.mean(dim=0)
                        pred_std_i_norm = samples_tensor.std(dim=0)
                
            # --- Denormalize Output ---
            if args.model.startswith('cnmp') or args.model.startswith('temp'):
                # Denormalize means back to workspace coordinates
                means_pred = denormalize_data(pred_mean_i_norm, Y_min_vals, Y_max_vals).cpu().numpy()
                
                # Standard deviation scales linearly with the range of the workspace
                y_range = (Y_max_vals - Y_min_vals).cpu().numpy()
                stds_pred = pred_std_i_norm.cpu().numpy() * y_range
            elif args.model.startswith('tedp'):
                # Denormalize means back to workspace coordinates
                all_samples_pred = denormalize_data(samples_tensor, Y_min_vals, Y_max_vals).cpu().numpy()
                
                # Apply Savitzky-Golay Filter to smooth the DDPM wiggles on all samples
                # window_length=15, polyorder=3 are standard values for 200-step trajectories
                all_samples_smoothed = savgol_filter(all_samples_pred, window_length=15, polyorder=3, axis=1)

            # --- Plotting ---
            dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
            
            for col_idx in range(full_dataset.d_y2):
                ax = axes[row_idx, col_idx]
                
                # Plot Ground Truth
                ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                        color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
                
                if args.model.startswith('cnmp') or args.model.startswith('temp'):
                    # Prediction
                    ax.plot(time_steps, means_pred[:, col_idx], 
                            color='blue', linestyle='--', linewidth=2, label='Pred')
                    
                    # Uncertainty
                    sigma = stds_pred[:, col_idx]
                    mean_curve = means_pred[:, col_idx]
                    ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                                    color='blue', alpha=0.1, label='Uncertainty')
                elif args.model.startswith('tedp'):
                    # Plot the "Uncertainty" Spaghetti (Background samples)
                    # We plot samples 1 through 9 with high transparency
                    for s_idx in range(1, num_mc_samples):
                        ax.plot(time_steps, all_samples_smoothed[s_idx, :, col_idx], 
                                color='blue', linestyle='-', linewidth=1, alpha=0.15)
                    
                    # Plot the Primary Representative Sample (Sample 0)
                    # This ensures we see one continuous, un-flattened trajectory
                    ax.plot(time_steps, all_samples_smoothed[0, :, col_idx], 
                            color='blue', linestyle='--', linewidth=2, label='Pred (Primary)')
                
                # Condition Points
                if mode['p'] == 2:
                    for idx in condition_points:
                        ax.scatter(time_steps[idx], curr_y_truth_raw[idx, col_idx], color='red', marker='o', s=80, label='Condition Point' if idx == condition_points[0] else "")

                # Labels
                if row_idx == 0:
                    ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=9, fontweight='bold')

                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize='small', loc='best')

        plt.suptitle(f"{mode['title']}\nModel: {args.model.upper()}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 
        
        save_file = f'{save_path}/prediction_{num_to_plot}_{mode["name"]}.png'
        plt.savefig(save_file)
        print(f"Prediction plots saved to {save_file}")

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load Data
    if args.dataset == "reassemble":
        base_data_folder = "data/paired_trajectories_insert_place"
    elif args.dataset == "synthetic_small":
        base_data_folder = "data/synthetic_trajectories"
    elif args.dataset == "synthetic_large":
        base_data_folder = "data/synthetic_trajectories_large"

    full_dataset = ReassembleDataset(data_dir=base_data_folder)

    if args.model == "cnmp":
        save_path = f"model/dual_cnmp_latent_alignment/save/{args.run_id}"
        from model.dual_cnmp_latent_alignment import dual_cnmp_model
        model = dual_cnmp_model.DualCNMP(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
    
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling"]:
        save_path = f"model/transformer_encoded_movement_primitive/save/{args.run_id}"
        if args.model == "temp_vanilla":
            from model.transformer_encoded_movement_primitive import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "temp_unmasked_pooling":
            from model.transformer_encoded_movement_primitive.unmasked_pooling import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"]:
        save_path = f"model/transformer_encoded_diffusion_policy/save/{args.run_id}"
        if args.model == "tedp_vanilla":
            from model.transformer_encoded_diffusion_policy import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "tedp_unmasked_pooling":
            from model.transformer_encoded_diffusion_policy.unmasked_pooling import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "tedp_cross_attention":
            from model.transformer_encoded_diffusion_policy.cross_attention_conditioning import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
    
    if not os.path.exists(save_path):
        print(f"Error: Save path {save_path} does not exist. Please check your run_id and ensure the model has been trained.")
        sys.exit(1)

    # Load the best model checkpoint
    checkpoint = torch.load(os.path.join(save_path, "best_model.pth"))
    if args.model == "cnmp":
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Normalize data
    if args.model == "cnmp":
        norm_stats = np.load(os.path.join(save_path, 'normalization_stats.npy'), allow_pickle=True).item()
    else:
        norm_stats = checkpoint['norm_stats']
    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']

    # Normalize data
    Y2_raw = full_dataset.Y2.clone()    # Keep a raw copy of Y2 for plotting ground truth
    full_dataset.Y1, full_dataset.Y2, full_dataset.C = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val)
    
    plot_grad_norms(save_path)

    if args.model == "cnmp":
        plot_training_progress_cnmp(save_path)
    else:
        plot_training_progress_temp_tedp(save_path)

    calculate_success_rates_and_plot(save_path, full_dataset, norm_stats, model, args, device=device)
    calculate_continuous_errors_and_plot(save_path, full_dataset, norm_stats, model, args, device=device)
    predict_random_trajectories(save_path, full_dataset, Y2_raw, norm_stats, model, args, num_samples=100, device=device)
