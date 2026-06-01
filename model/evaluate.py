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
    parser.add_argument("--model", type=str, required=True, choices=["cnmp", "temp_vanilla", "temp_unmasked_pooling", "temp_cls", "tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention", "tedp_cfg"], help="Which model architecture to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, choices=["reassemble", "synthetic_small", "synthetic_large"], help="Which dataset to evaluate on.")
    parser.add_argument("--run_id", type=str, required=True, help="Identifier for the model run to load and evaluate.")
    parser.add_argument("--fine_tuned", action='store_true', help="Perform evaluation on fine-tuned model.")
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

def plot_grad_norms(load_path, save_path, args):
    """Plots the gradient norms over epochs if they exist."""
    try:
        grad_norms = np.load(os.path.join(load_path, 'grad_norms.npy' if not args.fine_tuned else 'finetuning_grad_norms.npy'))
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

def plot_training_progress_cnmp(load_path, save_path, args):
    """Plots loss and error curves if they exist."""
    try:
        train_err = np.load(f'{load_path}/training_errors_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_training_errors_mse.npy')
        val_err = np.load(f'{load_path}/validation_errors_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_validation_errors_mse.npy')
        losses = np.load(f'{load_path}/losses_log_prob.npy' if not args.fine_tuned else f'{load_path}/finetuning_losses_log_prob.npy')

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
        plt.xlabel('Epoch (x400)')
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

def plot_training_progress_temp_tedp(load_path, save_path, args):
    """Plots loss and error curves if they exist."""
    try:
        composite_losses = np.load(f'{load_path}/composite_losses.npy' if not args.fine_tuned else f'{load_path}/finetuning_composite_losses.npy')
        train_fwd_mse = np.load(f'{load_path}/train_fwd_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_train_fwd_mse.npy')
        train_inv_mse = np.load(f'{load_path}/train_inv_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_train_inv_mse.npy')
        val_fwd_mse = np.load(f'{load_path}/val_fwd_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_val_fwd_mse.npy')
        val_inv_mse = np.load(f'{load_path}/val_inv_mse.npy' if not args.fine_tuned else f'{load_path}/finetuning_val_inv_mse.npy')

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(composite_losses, label='Composite Loss')
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

def calculate_success_rates_and_plot(load_path, save_path, full_dataset, norm_stats, model, args, device='cpu'):
    print("\n--- CALCULATING SUCCESS RATES & PLOTTING ---")

    Y_min_vals, Y_max_vals = norm_stats['Y_min'].to(device), norm_stats['Y_max'].to(device)
    global_range = (Y_max_vals - Y_min_vals).cpu().numpy()
    
    scenarios = [
        {'label': '5% (Strict)', 'thresh': 0.05 * global_range},
        {'label': '10% (Relaxed)', 'thresh': 0.10 * global_range}
    ]
    
    print(f"Global Range (X, Y, Z): {global_range}")
    
    time_len = full_dataset.time_len

    # Load the hidden test indices
    test_idx = np.load(os.path.join(load_path, 'test_indices.npy' if not args.fine_tuned else 'finetuning_test_indices.npy'))

    # Prepare the target time steps (all 200 points)
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
    
    # We now group strictly by Left Side (Train/Seen) vs Right Side (Extrapolation)
    groups = ['Left Side (Seen)', 'Right Side (Zero-Shot)']
    final_stats = {s['label']: {g: 0 for g in groups} for s in scenarios}
    group_counts = {g: 0 for g in groups}
    group_successes = {s['label']: {g: 0 for g in groups} for s in scenarios}

    print("Running Inference on Test Set...")
    for i in test_idx:
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)
        is_seen = full_dataset.valid_inverses[i]
        
        group_key = groups[0] if is_seen else groups[1]
        group_counts[group_key] += 1

        # Condition points
        condition_points = [0, -1] # t=0 and t=1 of the inverse trajectory corresponds to indices 0 and 199 (since time_len=200)
        eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) # Set all 200 to True (Masked)
        eval_mask[0, condition_points] = False # Set condition points to False (Observed)
        
        # Run Inference
        with torch.no_grad():
            if args.model.startswith('cnmp'):
                cond_pts = [[x_full[0, idx], y2_seq[0, idx]] for idx in condition_points]
                pred_seq, _ = model_predict.predict_inverse_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
            elif args.model.startswith('temp'):
                output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
                _, _, pred_seq, _ = output.chunk(4, dim=-1)
            elif args.model.startswith('tedp'):
                pred_seq = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
        
        # Denormalize Predictions and Ground Truth to physical scale
        pred_traj = denormalize_data(pred_seq.squeeze(0), Y_min_vals, Y_max_vals).cpu().numpy()
        gt_traj = denormalize_data(full_dataset.Y2[i].to(device), Y_min_vals, Y_max_vals).cpu().numpy()
        
        pred_start, gt_start = pred_traj[0], gt_traj[0]
        pred_end, gt_end = pred_traj[-1], gt_traj[-1]

        for s in scenarios:
            start_ok = np.all(np.abs(pred_start - gt_start)[:2] <= s['thresh'][:2])
            end_ok = np.all(np.abs(pred_end - gt_end)[:2] <= s['thresh'][:2])
            if start_ok and end_ok:
                group_successes[s['label']][group_key] += 1

    # Calculate Rates
    for s in scenarios:
        for g in groups:
            if group_counts[g] > 0:
                final_stats[s['label']][g] = (group_successes[s['label']][g] / group_counts[g]) * 100

    # Plot Bar Chart
    labels = [f"{g}\n(n={group_counts[g]})" for g in groups]
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [final_stats['5% (Strict)'][g] for g in groups], width, label='5% Tolerance', color='#d9534f')
    rects2 = ax.bar(x + width/2, [final_stats['10% (Relaxed)'][g] for g in groups], width, label='10% Tolerance', color='#5bc0de')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Spatial Generalization Success Rates\n{args.model.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'success_rate_extrapolation.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Bar chart saved to {plot_path}")

def calculate_continuous_errors_and_plot(load_path, save_path, full_dataset, norm_stats, model, args, device='cpu'):
    print("\n--- CALCULATING CONTINUOUS ERRORS (CM) & PLOTTING ---")

    Y_min_vals, Y_max_vals = norm_stats['Y_min'].to(device), norm_stats['Y_max'].to(device)
    time_len = full_dataset.time_len
    
    # Load the hidden test indices
    test_idx = np.load(os.path.join(load_path, 'test_indices.npy' if not args.fine_tuned else 'finetuning_test_indices.npy'))
    print(f"Evaluating continuous errors on {len(test_idx)} test samples...")
    
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)

    metrics = ['Euclidean (3D)', 'X-Axis (Left/Right)', 'Y-Axis (Forward/Back)', 'Z-Axis (Depth)']
    groups = ['Left Side (Seen)', 'Right Side (Zero-Shot)']
    
    start_errors = {m: {g: [] for g in groups} for m in metrics}
    end_errors = {m: {g: [] for g in groups} for m in metrics}

    for i in test_idx:
        y1_seq = full_dataset.Y1[i].unsqueeze(0).to(device)
        y2_seq = full_dataset.Y2[i].unsqueeze(0).to(device)
        curr_context = full_dataset.C[i].view(1, 1, -1).to(device)
        is_seen = full_dataset.valid_inverses[i]
        group_key = groups[0] if is_seen else groups[1]

        condition_points = [0, -1] 
        eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device)
        eval_mask[0, condition_points] = False 
        
        # Run Inference
        with torch.no_grad():
            if args.model.startswith('cnmp'):
                cond_pts = [[x_full[0, idx], y2_seq[0, idx]] for idx in condition_points]
                pred_seq, _ = model_predict.predict_inverse_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
            elif args.model.startswith('temp'):
                output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
                _, _, pred_seq, _ = output.chunk(4, dim=-1)
            elif args.model.startswith('tedp'):
                pred_seq = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
        
        pred_traj = denormalize_data(pred_seq.squeeze(0), Y_min_vals, Y_max_vals).cpu().numpy()
        gt_traj = denormalize_data(full_dataset.Y2[i].to(device), Y_min_vals, Y_max_vals).cpu().numpy()
        
        # Calculate Differences in cm
        diff_start = (pred_traj[0, :3] - gt_traj[0, :3]) * 100.0
        diff_end = (pred_traj[-1, :3] - gt_traj[-1, :3]) * 100.0
        
        start_errors['Euclidean (3D)'][group_key].append(np.linalg.norm(diff_start))
        start_errors['X-Axis (Left/Right)'][group_key].append(abs(diff_start[0]))
        start_errors['Y-Axis (Forward/Back)'][group_key].append(abs(diff_start[1]))
        start_errors['Z-Axis (Depth)'][group_key].append(abs(diff_start[2]))
        
        end_errors['Euclidean (3D)'][group_key].append(np.linalg.norm(diff_end))
        end_errors['X-Axis (Left/Right)'][group_key].append(abs(diff_end[0]))
        end_errors['Y-Axis (Forward/Back)'][group_key].append(abs(diff_end[1]))
        end_errors['Z-Axis (Depth)'][group_key].append(abs(diff_end[2]))

    def create_violin_figure(error_data, time_title, filename):
        rows = []
        for m in metrics:
            for g in groups:
                for val in error_data[m][g]:
                    rows.append({'Metric': m, 'Domain': g, 'Error (cm)': val})
        df = pd.DataFrame(rows)

        # Save the error data for further plotting
        csv_filename = filename.replace('.png', '.csv')
        df.to_csv(os.path.join(save_path, csv_filename), index=False)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        fig.suptitle(f'Trajectory {time_title} Deviation\nExtrapolation Check', fontsize=16, fontweight='bold', y=0.98)

        for idx, m in enumerate(metrics):
            ax = axes[idx]
            df_metric = df[df['Metric'] == m]
            sns.violinplot(
                data=df_metric, 
                x='Domain', 
                y='Error (cm)', 
                ax=ax, 
                hue='Domain',
                palette=['#5bc0de', '#d9534f'], 
                inner='box', 
                legend=False,
                cut=0,   # Prevent the violin from drawing density below 0 cm
                density_norm='width'
            )
            ax.set_title(m, fontsize=14, fontweight='bold')
            ax.set_ylabel('Error (cm)', fontsize=12, fontweight='bold')
            ax.set_xlabel('') 
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Saved: {plot_path}")

    create_violin_figure(start_errors, "Start Point (t=0)", 'continuous_error_violins_start_extrap.png')
    create_violin_figure(end_errors, "End Point (t=1)", 'continuous_error_violins_end_extrap.png')

def predict_random_trajectories(load_path, save_path, full_dataset, Y2_raw, norm_stats, model, args, num_samples=6, device='cpu'):
    print(f"\n--- PREDICTING RANDOM TRAJECTORIES ({num_samples} samples) ---")

    Y_min_vals, Y_max_vals = norm_stats['Y_min'].to(device), norm_stats['Y_max'].to(device)
    C_min_val, C_max_val = norm_stats['C_min'].to(device), norm_stats['C_max'].to(device)
    time_len = full_dataset.time_len

    test_idx = np.load(os.path.join(load_path, 'test_indices.npy' if not args.fine_tuned else 'finetuning_test_indices.npy')).tolist()
    
    # Stratified Sampling: Grab half from Seen, half from Zero-Shot if possible
    seen_idx = [i for i in test_idx if full_dataset.valid_inverses[i]]
    unseen_idx = [i for i in test_idx if not full_dataset.valid_inverses[i]]
    
    sample_seen = random.sample(seen_idx, min(len(seen_idx), num_samples//2))
    sample_unseen = random.sample(unseen_idx, min(len(unseen_idx), num_samples - len(sample_seen)))
    indices = sample_seen + sample_unseen
    num_to_plot = len(indices)
    
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
            # 1. Denormalize the Task Parameter to display the actual Goal Coordinates
            curr_context = full_dataset.C[traj_idx].view(1, 1, -1).to(device)
            denorm_context = denormalize_data(curr_context.squeeze(), C_min_val, C_max_val).cpu().numpy()
            goal_x, goal_y = denorm_context[0], denorm_context[1]
            
            is_seen = full_dataset.valid_inverses[traj_idx]
            domain_label = "Seen Domain" if is_seen else "Zero-Shot Extrapolation"
            curr_obj_name = f"Goal:\nX={goal_x:.2f}, Y={goal_y:.2f}\n({domain_label})"
            
            curr_y_truth_raw = Y2_raw[traj_idx].numpy()
            y1_seq = full_dataset.Y1[traj_idx].unsqueeze(0).to(device)
            y2_seq = full_dataset.Y2[traj_idx].unsqueeze(0).to(device)

            samples = []
            with torch.no_grad():
                if mode['p'] == 1:
                    # Condition on the Forward Trajectory
                    condition_points = [60]
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
                    eval_mask[0, condition_points] = False 

                    if args.model.startswith('cnmp'):
                        cond_pts = [[x_full[0, idx], y1_seq[0, idx]] for idx in condition_points]
                        pred_mean_i_norm, pred_std_i_norm = model_predict.predict_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                    elif args.model.startswith('temp'):
                        output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_1=eval_mask) # The p=mode['p'] parameter forces the decoder to use the correct latent vector
                        _, _, pred_mean_i_norm, pred_std_i_norm = output.chunk(4, dim=-1) # Extract the Inverse Trajectory predictions (Mean and Log-Variance)
                        pred_std_i_norm = torch.log(1 + torch.exp(pred_std_i_norm)) # Convert log variance parameter to standard deviation
                        pred_mean_i_norm, pred_std_i_norm = pred_mean_i_norm.squeeze(0), pred_std_i_norm.squeeze(0)
                    elif args.model.startswith('tedp'):
                        num_mc_samples = 6 
                        for _ in range(num_mc_samples):
                            pred = model.sample(cond_seq=y1_seq, params=curr_context, mask_indices=eval_mask, source_dim='y1', target_dim='y2', time_len=time_len)
                            samples.append(pred.squeeze(0))
                        samples_tensor = torch.stack(samples)
                        pred_mean_i_norm, pred_std_i_norm = samples_tensor.mean(dim=0), samples_tensor.std(dim=0)
                else:
                    # Condition on Inverse Trajectory
                    condition_points = [0, -1] 
                    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
                    eval_mask[0, condition_points] = False 

                    if args.model.startswith('cnmp'):
                        cond_pts = [[x_full[0, idx], y2_seq[0, idx]] for idx in condition_points]
                        pred_mean_i_norm, pred_std_i_norm = model_predict.predict_inverse_inverse(model, time_len, curr_context, cond_pts, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                    elif args.model.startswith('temp'):
                        output, _, _, _ = model(y1_seq, y2_seq, curr_context, x_full, extra_pass=False, p=mode['p'], mask_indices_2=eval_mask)
                        _, _, pred_mean_i_norm, pred_std_i_norm = output.chunk(4, dim=-1) # Extract the Inverse Trajectory predictions (Mean and Log-Variance)
                        pred_std_i_norm = torch.log(1 + torch.exp(pred_std_i_norm)) 
                        pred_mean_i_norm, pred_std_i_norm = pred_mean_i_norm.squeeze(0), pred_std_i_norm.squeeze(0)
                    elif args.model.startswith('tedp'):
                        num_mc_samples = 6
                        for _ in range(num_mc_samples):
                            pred = model.sample(cond_seq=y2_seq, params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
                            samples.append(pred.squeeze(0))
                        samples_tensor = torch.stack(samples) 
                        pred_mean_i_norm, pred_std_i_norm = samples_tensor.mean(dim=0), samples_tensor.std(dim=0)

            # --- Denormalize Output ---
            if args.model.startswith('cnmp') or args.model.startswith('temp'):
                # Denormalize means back to workspace coordinates
                means_pred = denormalize_data(pred_mean_i_norm, Y_min_vals, Y_max_vals).cpu().numpy()
                
                # Standard deviation scales linearly with the range of the workspace
                y_range = (Y_max_vals - Y_min_vals).cpu().numpy()
                stds_pred = pred_std_i_norm.cpu().numpy() * y_range
            elif args.model.startswith('tedp'):
                all_samples_pred = denormalize_data(samples_tensor, Y_min_vals, Y_max_vals).cpu().numpy()

                # Apply Savitzky-Golay Filter to smooth the DDPM wiggles on all samples
                all_samples_smoothed = savgol_filter(all_samples_pred, window_length=15, polyorder=3, axis=1)

            # --- Plotting ---
            dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
            for col_idx in range(full_dataset.d_y2):
                ax = axes[row_idx, col_idx]
                ax.plot(time_steps, curr_y_truth_raw[:, col_idx], color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
                
                if args.model.startswith('cnmp') or args.model.startswith('temp'):
                    # Prediction
                    ax.plot(time_steps, means_pred[:, col_idx], color='blue', linestyle='--', linewidth=2, label='Pred')
                    
                    # Uncertainty
                    sigma = stds_pred[:, col_idx]
                    mean_curve = means_pred[:, col_idx]
                    ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, color='blue', alpha=0.1, label='Uncertainty')
                elif args.model.startswith('tedp'):
                    # Plot the "Uncertainty" Spaghetti (Background samples)
                    # We plot samples 1 through 9 with high transparency
                    for s_idx in range(1, num_mc_samples):
                        ax.plot(time_steps, all_samples_smoothed[s_idx, :, col_idx], color='blue', linestyle='-', linewidth=1, alpha=0.15)

                    # Plot the Primary Representative Sample (Sample 0)
                    # This ensures we see one continuous, un-flattened trajectory
                    ax.plot(time_steps, all_samples_smoothed[0, :, col_idx], color='blue', linestyle='--', linewidth=2, label='Pred (Primary)')
                
                # Condition Points
                if mode['p'] == 2:
                    for idx in condition_points:
                        ax.scatter(time_steps[idx], curr_y_truth_raw[idx, col_idx], color='red', marker='o', s=80, label='Condition Point' if idx == condition_points[0] else "")

                # Labels
                if row_idx == 0:
                    ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{curr_obj_name}\nPair {traj_idx}", fontsize=10, fontweight='bold')

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
        load_path = f"model/dual_cnmp_latent_alignment/save/{args.run_id}"
        from model.dual_cnmp_latent_alignment import dual_cnmp_model
        model = dual_cnmp_model.DualCNMP(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
    elif args.model in ["temp_vanilla", "temp_unmasked_pooling", "temp_cls"]:
        load_path = f"model/transformer_encoded_movement_primitive/save/{args.run_id}"
        if args.model == "temp_vanilla":
            from model.transformer_encoded_movement_primitive import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "temp_unmasked_pooling":
            from model.transformer_encoded_movement_primitive.unmasked_pooling import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "temp_cls":
            from model.transformer_encoded_movement_primitive.cls_token import temp_model
            model = temp_model.TempModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
    elif args.model in ["tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention", "tedp_cfg"]:
        load_path = f"model/transformer_encoded_diffusion_policy/save/{args.run_id}"
        if args.model == "tedp_vanilla":
            from model.transformer_encoded_diffusion_policy import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "tedp_unmasked_pooling":
            from model.transformer_encoded_diffusion_policy.unmasked_pooling import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "tedp_cross_attention":
            from model.transformer_encoded_diffusion_policy.cross_attention_conditioning import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
        elif args.model == "tedp_cfg":
            from model.transformer_encoded_diffusion_policy.classifier_free_guidance import tedp_model
            model = tedp_model.TedpModel(full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, full_dataset.d_param).to(device)
    
    if not os.path.exists(load_path):
        print(f"Error: Load path {load_path} does not exist. Please check your run_id and ensure the model has been trained.")
        sys.exit(1)

    save_path = os.path.join(load_path, "pretrained" if not args.fine_tuned else "finetuned")
    os.makedirs(save_path, exist_ok=True)

    # Load the best model checkpoint
    checkpoint = torch.load(os.path.join(load_path, "best_model.pth" if not args.fine_tuned else "finetuning_best_model.pth"))
    if args.model == "cnmp":
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Normalize data
    if args.model == "cnmp":
        norm_stats = np.load(os.path.join(load_path, 'normalization_stats.npy'), allow_pickle=True).item()
    else:
        norm_stats = checkpoint['norm_stats']
    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']

    # Normalize data
    Y2_raw = full_dataset.Y2.clone()    # Keep a raw copy of Y2 for plotting ground truth
    full_dataset.Y1, full_dataset.Y2, full_dataset.C = normalize_data(full_dataset.Y1, full_dataset.Y2, full_dataset.C, Y_min_vals, Y_max_vals, C_min_val, C_max_val)
    
    plot_grad_norms(load_path, save_path, args)

    if args.model == "cnmp":
        plot_training_progress_cnmp(load_path, save_path, args)
    else:
        plot_training_progress_temp_tedp(load_path, save_path, args)

    calculate_success_rates_and_plot(load_path, save_path, full_dataset, norm_stats, model, args, device=device)
    calculate_continuous_errors_and_plot(load_path, save_path, full_dataset, norm_stats, model, args, device=device)
    predict_random_trajectories(load_path, save_path, full_dataset, Y2_raw, norm_stats, model, args, num_samples=50, device=device)
