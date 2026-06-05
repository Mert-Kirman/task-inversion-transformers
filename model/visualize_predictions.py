from datetime import datetime
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import ReassembleDataset
from model.dual_cnmp_latent_alignment import dual_cnmp_model
from model.transformer_encoded_movement_primitive.cls_token import temp_model
from model.transformer_encoded_diffusion_policy.classifier_free_guidance import tedp_model
import model.model_predict as model_predict
from model.utils import seed_everything

from scipy.signal import savgol_filter

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D visualizations of model predictions.")
    parser.add_argument("--num_plots", type=int, default=10, help="Number of samples to visualize.")
    parser.add_argument("--dataset_path", type=str, default="data/paired_trajectories_insert_place")
    
    parser.add_argument("--cnmp_run", type=str, default="model/dual_cnmp_latent_alignment/save/run_20260601_002126")
    parser.add_argument("--temp_run", type=str, default="model/transformer_encoded_movement_primitive/save/run_20260601_001803")
    parser.add_argument("--tedp_run", type=str, default="model/transformer_encoded_diffusion_policy/save/run_20260601_001323")
    
    parser.add_argument("--fine_tuned", action='store_true', help="Whether to load fine-tuned models (if not set, will load pre-fine-tuning checkpoints).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default="model/visualizations/prediction_comparisons_3d", help="Directory to save the generated 3D plots.")
    return parser.parse_args()

def draw_table_and_origin(ax, stats):
    """Draws a brown rectangle for the table and a star for the origin."""
    Y_min = stats['Y_min'].cpu().numpy()
    Y_max = stats['Y_max'].cpu().numpy()
    
    # Create the table surface bounds (expanding slightly beyond the min/max X/Y for aesthetics)
    x_min, x_max = Y_min[0] - 0.05, Y_max[0] + 0.05
    y_min, y_max = Y_min[1] - 0.05, Y_max[1] + 0.05
    z_table = 0.0 # Assuming the table is at Z=0

    # Draw the table plane
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.zeros_like(xx) + z_table
    
    ax.plot_surface(xx, yy, zz, color='saddlebrown', alpha=0.15, edgecolor='none')
    
    # Mark the Origin / Target Goal
    ax.scatter(0, 0, 0, color='red', marker='*', s=300, label='Origin (Target Base)', zorder=5)

def main():
    args = parse_args()
    seed_everything(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_dir, f"{run_id}")
    os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading Reassemble Dataset...")
    dataset = ReassembleDataset(args.dataset_path)
    
    # 2. Load Normalization Stats & Test Indices
    stats_path = os.path.join(args.cnmp_run, 'normalization_stats.npy')
    test_idx_path = os.path.join(args.cnmp_run, 'finetuning_test_indices.npy' if args.fine_tuned else 'test_indices.npy')
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Could not find {stats_path}. Please check your run folder paths.")
        
    stats = np.load(stats_path, allow_pickle=True).item()
    test_indices = np.load(test_idx_path)
    
    Y_min = stats['Y_min'].to(device)
    Y_max = stats['Y_max'].to(device)
    C_min = stats['C_min'].to(device)
    C_max = stats['C_max'].to(device)
    epsilon = 1e-8

    # 3. Load Models
    print("Loading Models...")
    # CNMP
    cnmp = dual_cnmp_model.DualCNMP(dataset.d_x, dataset.d_y1, dataset.d_y2, dataset.d_param).to(device)
    cnmp.load_state_dict(torch.load(os.path.join(args.cnmp_run, 'finetuning_best_model.pth' if args.fine_tuned else 'best_model.pth'), map_location=device))
    cnmp.eval()

    # TEMP
    temp = temp_model.TempModel(dataset.d_x, dataset.d_y1, dataset.d_y2, dataset.d_param).to(device)
    temp.load_state_dict(torch.load(os.path.join(args.temp_run, 'finetuning_best_model.pth' if args.fine_tuned else 'best_model.pth'), map_location=device)['model_state_dict'])
    temp.eval()

    # TEDP
    tedp = tedp_model.TedpModel(dataset.d_x, dataset.d_y1, dataset.d_y2, dataset.d_param).to(device)
    tedp.load_state_dict(torch.load(os.path.join(args.tedp_run, 'finetuning_best_model.pth' if args.fine_tuned else 'best_model.pth'), map_location=device)['model_state_dict'])
    tedp.eval()

    # Filter to only paired test samples
    paired_test_indices = [idx for idx in test_indices if dataset.valid_inverses[idx]]
    selected_indices = np.random.choice(paired_test_indices, min(args.num_plots, len(paired_test_indices)), replace=False)

    print(f"Generating {len(selected_indices)} plots...")

    for plot_idx, sample_idx in enumerate(selected_indices):
        print(f"Processing Sample {plot_idx + 1}/{len(selected_indices)} (Dataset Index: {sample_idx})...")
        
        # Extract and Normalize Data
        Y1 = dataset.Y1[sample_idx:sample_idx+1].to(device)
        Y2 = dataset.Y2[sample_idx:sample_idx+1].to(device)
        C = dataset.C[sample_idx:sample_idx+1].to(device)

        Y1_norm = (Y1 - Y_min) / (Y_max - Y_min + epsilon)
        Y2_norm = (Y2 - Y_min) / (Y_max - Y_min + epsilon)
        C_norm = (C - C_min) / (C_max - C_min + epsilon)
        C_norm = C_norm.unsqueeze(1)

        time_len = dataset.time_len
        x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)

        with torch.no_grad():
            # Condition on Inverse Trajectory
            condition_points = [0, -1] 
            eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
            eval_mask[0, condition_points] = False 

            # --- 1. CNMP Inference ---
            # Prepare Condition
            cond_pts = []
            for idx in condition_points:
                cond_pts.append([x_full[0, idx], Y2_norm[0, idx]])

            pred_seq_cnmp, _ = model_predict.predict_inverse_inverse(cnmp, time_len, C_norm, cond_pts, dataset.d_x, dataset.d_y1, dataset.d_y2, device=device)
            
            # --- 2. TEMP Inference ---
            output, _, _, _ = temp(Y1_norm, Y2_norm, C_norm, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
            _, _, pred_seq, _ = output.chunk(4, dim=-1)
            pred_seq_temp = pred_seq.squeeze(0)
            
            # --- 3. TEDP Inference ---
            pred_seq_tedp = tedp.sample(cond_seq=Y2_norm, params=C_norm, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
            pred_seq_tedp = pred_seq_tedp.squeeze(0)

        # Un-normalize for plotting
        def unnorm(tensor):
            return (tensor.cpu().squeeze(0).numpy() * (Y_max.cpu().numpy() - Y_min.cpu().numpy())) + Y_min.cpu().numpy()

        gt_inv = unnorm(Y2_norm)
        pred_cnmp = unnorm(pred_seq_cnmp)
        pred_temp = unnorm(pred_seq_temp)
        pred_tedp = unnorm(pred_seq_tedp)
        pred_tedp = savgol_filter(pred_tedp, window_length=15, polyorder=3, axis=0)

        # --- PLOTTING ---
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Draw Environment
        draw_table_and_origin(ax, stats)

        # Plot Ground Truth
        ax.plot(gt_inv[:, 0], gt_inv[:, 1], gt_inv[:, 2], color='black', linestyle='--', linewidth=2, alpha=0.6, label='Ground Truth')
        
        # Plot Condition Point
        ax.scatter(*gt_inv[0], color='black', s=100, marker='o', edgecolors='white', zorder=10, label='Condition Point (t=0)')

        # Plot Predictions
        ax.plot(pred_cnmp[:, 0], pred_cnmp[:, 1], pred_cnmp[:, 2], color='#1f77b4', linewidth=2.5, label='CNMP Prediction')
        ax.plot(pred_temp[:, 0], pred_temp[:, 1], pred_temp[:, 2], color='#ff7f0e', linewidth=2.5, label='TEMP Prediction')
        ax.plot(pred_tedp[:, 0], pred_tedp[:, 1], pred_tedp[:, 2], color='#2ca02c', linewidth=2.5, label='TEDP Prediction')

        # Aesthetics
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f"Trajectory Inversion Comparison\nTest Sample #{sample_idx}")
        
        # Legend (place outside plot if it gets cluttered)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # Set stable view angle to see the table and the arc clearly
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        save_path = os.path.join(save_folder, f"comparison_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nAll plots saved to '{save_folder}'.")

if __name__ == "__main__":
    main()
