import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Evaluate conditioning of CNMP/TEMP/TEDP models on perturbed Reassemble/Synthetic datasets to check conditioning response.")
    parser.add_argument("--model", type=str, required=True, choices=["cnmp", "temp_vanilla", "temp_unmasked_pooling", "tedp_vanilla", "tedp_unmasked_pooling", "tedp_cross_attention"], help="Which model architecture to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, choices=["reassemble", "synthetic_small", "synthetic_large"], help="Which dataset to evaluate conditioning on.")
    parser.add_argument("--run_id", type=str, required=True, help="Identifier for the model run to load and evaluate.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trajectories to sample from the test set for evaluation.")
    parser.add_argument("--perturb_pct", type=float, default=0.10, help="Percentage to shift the conditioning points (e.g., 0.10 for 10%).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args

def normalize_data(tensor, min_val, max_val):
    """Normalize a tensor using global min-max normalization based on training data statistics."""
    epsilon = 1e-8
    tensor_normalized = (tensor - min_val) / (max_val - min_val + epsilon)

    return tensor_normalized

def denormalize_data(tensor, min_val, max_val):
    """Reverts [0, 1] data back to original scale."""
    denominator = max_val - min_val
    return tensor * denominator + min_val

def condition_perturbation_test(save_path, full_dataset, Y2_raw, norm_stats, model, args, device='cpu'):
    print(f"\n--- RUNNING CONDITIONING PERTURBATION TEST ({args.num_samples} samples) ---")

    Y_min_vals, Y_max_vals, C_min_val, C_max_val = norm_stats['Y_min'], norm_stats['Y_max'], norm_stats['C_min'], norm_stats['C_max']

    time_len = full_dataset.time_len

    # Move bounds to device for denormalization
    Y_min_vals = Y_min_vals.to(device)
    Y_max_vals = Y_max_vals.to(device)
    C_min_val = C_min_val.to(device)
    C_max_val = C_max_val.to(device)

    # Load test indices and sample from them
    test_idx = np.load(os.path.join(save_path, 'test_indices.npy'))
    test_idx_list = test_idx.tolist()
    
    num_to_plot = min(args.num_samples, len(test_idx_list))
    indices = random.sample(test_idx_list, num_to_plot)
    
    time_steps = np.linspace(0, 1, time_len)
    print(f"Evaluating indices: {indices}")

    # Set up the plot
    fig, axes = plt.subplots(num_to_plot, full_dataset.d_y2, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    # Condition on t=0 and t=1
    cond_step_indices = [0, -1] 
    eval_mask = torch.ones(1, time_len, dtype=torch.bool, device=device) 
    eval_mask[0, cond_step_indices] = False # False = Observed

    for row_idx, traj_idx in enumerate(indices):
        # Identify Object Type
        curr_context_norm = full_dataset.C[traj_idx]
        raw_id = denormalize_data(curr_context_norm.to(device), C_min_val, C_max_val)[-1].item()
        
        curr_obj_name = "Unknown"
        min_diff = float('inf')
        for key, config in full_dataset.object_config.items():
            if abs(config['id'] - raw_id) < min_diff:
                min_diff = abs(config['id'] - raw_id)
                curr_obj_name = config['label']
        
        # --- Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() 
        curr_context = full_dataset.C[traj_idx].view(1, 1, -1).to(device)

        # --- Prepare Perturbed Sequences ---
        y2_seq_orig = full_dataset.Y2[traj_idx].unsqueeze(0).to(device)
        y2_seq_minus = y2_seq_orig.clone()
        y2_seq_plus = y2_seq_orig.clone()

        condition_sets = {
            "Original": {"seq": y2_seq_orig, "points": []},
            f"Shifted -{int(args.perturb_pct * 100)}%": {"seq": y2_seq_minus, "points": []},
            f"Shifted +{int(args.perturb_pct * 100)}%": {"seq": y2_seq_plus, "points": []}
        }

        # Apply shifts directly to the condition tensors
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            
            val_orig_norm = y2_seq_orig[0, t_idx, :].clone()

            if t_idx == -1:
                val_minus_norm = val_orig_norm
                val_plus_norm = val_orig_norm
            else:
                val_minus_norm = val_orig_norm - args.perturb_pct
                val_plus_norm = val_orig_norm + args.perturb_pct
            
            y2_seq_minus[0, t_idx, :] = val_minus_norm
            y2_seq_plus[0, t_idx, :] = val_plus_norm
            
            # Denormalize strictly for plotting the condition X's
            val_orig_raw = denormalize_data(val_orig_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            val_minus_raw = denormalize_data(val_minus_norm, Y_min_vals, Y_max_vals).cpu().numpy()
            val_plus_raw = denormalize_data(val_plus_norm, Y_min_vals, Y_max_vals).cpu().numpy()

            condition_sets["Original"]["points"].append({"t": t_val, "raw": val_orig_raw})
            condition_sets[f"Shifted -{int(args.perturb_pct * 100)}%"]["points"].append({"t": t_val, "raw": val_minus_raw})
            condition_sets[f"Shifted +{int(args.perturb_pct * 100)}%"]["points"].append({"t": t_val, "raw": val_plus_raw})

        # --- Run Inference ---
        predictions = {}
        for cond_name, cond_data in condition_sets.items():
            with torch.no_grad():
                if args.model.startswith('cnmp'):
                    # Extract just the [time, normalized_value] pair the model expects
                    model_cond = [[c["t"], normalize_data(torch.from_numpy(c["raw"]).to(device), Y_min_vals, Y_max_vals)] for c in cond_data["points"]]
                    
                    pred_seq, _ = model_predict.predict_inverse_inverse(model, time_len, curr_context, model_cond, full_dataset.d_x, full_dataset.d_y1, full_dataset.d_y2, device=device)
                elif args.model.startswith('temp'):
                    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
                    y1_seq = full_dataset.Y1[traj_idx].unsqueeze(0).to(device)
                    
                    output, _, _, _ = model(y1_seq, cond_data["seq"], curr_context, x_full, extra_pass=False, p=2, mask_indices_2=eval_mask)
                    _, _, pred_seq, _ = output.chunk(4, dim=-1)
                    pred_seq = pred_seq.squeeze(0)
                elif args.model.startswith('tedp'):
                    pred_seq = model.sample(cond_seq=cond_data["seq"], params=curr_context, mask_indices=eval_mask, source_dim='y2', target_dim='y2', time_len=time_len)
                    pred_seq = pred_seq.squeeze(0)

            # Denormalize
            pred_seq_raw = denormalize_data(pred_seq, Y_min_vals, Y_max_vals).cpu().numpy()
            
            if args.model.startswith('tedp'):
                # Apply Savitzky-Golay Filter to smooth the DDPM wiggles on TEDP predictions
                pred_seq_raw = savgol_filter(pred_seq_raw, window_length=15, polyorder=3, axis=0)

            predictions[cond_name] = pred_seq_raw

        # --- Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        colors = {"Original": "blue", f"Shifted -{int(args.perturb_pct * 100)}%": "orange", f"Shifted +{int(args.perturb_pct * 100)}%": "green"}
        
        for col_idx in range(full_dataset.d_y2):
            ax = axes[row_idx, col_idx]
            
            # Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # Plot all Predictions and their corresponding Condition Points
            for cond_name, pred_data in predictions.items():
                c_color = colors[cond_name]
                
                # Plot Predicted Line
                ax.plot(time_steps, pred_data[:, col_idx], color=c_color, linestyle='--', linewidth=2, label=f'Pred ({cond_name})')
                
                # Plot the Condition Points using the RAW coordinate
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

    plt.suptitle(f"Conditioning Perturbation Test\nModel: {args.model.upper()}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/conditioning_perturbation_{int(args.perturb_pct * 100)}_pct.png'
    plt.savefig(save_file)
    print(f"Conditioning perturbation plots saved to {save_file}")


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
    full_dataset.Y1 = normalize_data(full_dataset.Y1, Y_min_vals, Y_max_vals)
    full_dataset.Y2 = normalize_data(full_dataset.Y2, Y_min_vals, Y_max_vals)
    full_dataset.C = normalize_data(full_dataset.C, C_min_val, C_max_val)

    condition_perturbation_test(save_path, full_dataset, Y2_raw, norm_stats, model, args, device=device)
