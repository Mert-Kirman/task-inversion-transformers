import os
import sys
import torch
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import ReassembleDataset
# Assuming you want to test the Unmasked Pooling variant
from model.transformer_encoded_movement_primitive.unmasked_pooling import temp_model

def find_test_pair(dataset):
    """
    Finds two trajectories with nearly identical Task Parameters (C)
    but physically distinct starting points for their INVERSE trajectories (Y2 at t=0).
    """
    print("Searching for an ideal test pair (Same Task, Different Inverse Start)...")
    C = dataset.C
    Y2 = dataset.Y2
    
    num_samples = len(C)
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            # 1. Must be the same Object ID
            if C[i][2] != C[j][2]:
                continue
                
            # 2. The Task Parameters (avg_x, avg_y, Object ID) must be nearly identical
            c_dist = torch.norm(C[i] - C[j]).item()
            
            # 3. The START of the INVERSE trajectory must be physically different
            # (This represents a different Hungarian matching gap / extraction point)
            start_dist = torch.norm(Y2[i, 0, :3] - Y2[j, 0, :3]).item()
            
            # Thresholds: C_dist < 0.02 (Very similar context), start_dist > 0.03 (At least 3cm apart at t=0)
            if c_dist < 0.02: print(f"Potential Match Found (Contextually Similar): C_dist={c_dist:.4f}")
            if start_dist > 0.03: print(f"Potential Match Found (Different Inverse Start): Start_dist={start_dist:.4f}")
            if c_dist < 0.02 and start_dist > 0.03:
                print(f"Match Found! Indices: {i} and {j}")
                print(f"  Task Param Distance: {c_dist:.4f}")
                print(f"  Inverse Start Point Distance: {start_dist:.4f} meters")
                return i, j
                
    raise ValueError("Could not find a pair matching the strict criteria. Try loosening the thresholds.")

def evaluate_conditioning_response(run_folder, dataset_path="data/paired_trajectories_insert_place"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset & Normalization Stats
    print("Loading Dataset...")
    dataset = ReassembleDataset(dataset_path)
    
    checkpoint = torch.load(os.path.join(run_folder, "best_model.pth"))
    norm_stats = checkpoint['norm_stats']
    Y_min_vals = norm_stats['Y_min'].cpu()
    Y_max_vals = norm_stats['Y_max'].cpu()
    C_min_val = norm_stats['C_min'].cpu()
    C_max_val = norm_stats['C_max'].cpu()
    
    # 2. Find the perfect isolation pair
    idx_A, idx_B = find_test_pair(dataset)
    
    # Normalize the specific samples
    epsilon = 1e-8
    Y1_A = (dataset.Y1[idx_A:idx_A+1] - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2_A = (dataset.Y2[idx_A:idx_A+1] - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    
    Y1_B = (dataset.Y1[idx_B:idx_B+1] - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2_B = (dataset.Y2[idx_B:idx_B+1] - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    
    # THE ISOLATION TRICK: We use Task Parameter A for BOTH inferences!
    C_A = (dataset.C[idx_A:idx_A+1] - C_min_val) / (C_max_val - C_min_val + epsilon)
    
    # Move to device
    Y1_A, Y1_B = Y1_A.to(device), Y1_B.to(device)
    Y2_A, Y2_B = Y2_A.to(device), Y2_B.to(device)
    C_A = C_A.unsqueeze(1).to(device)
    
    # 3. Load Model
    print("Loading Pre-Trained Model...")
    model = temp_model.TempModel(dataset.d_x, dataset.d_y1, dataset.d_y2, dataset.d_param, dropout_p=[0.0, 0.0]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Setup Inference Mask (Observe ONLY t=0 of the INVERSE trajectory)
    time_len = dataset.time_len
    x_full = torch.linspace(0, 1, time_len, device=device).view(1, time_len, 1)
    
    val_mask_1 = torch.ones(1, time_len, dtype=torch.bool, device=device) # Forward is completely masked
    val_mask_2 = torch.ones(1, time_len, dtype=torch.bool, device=device)
    val_mask_2[0, 0] = False # Unmask ONLY t=0 of the inverse trajectory
    
    # 5. Run Inference
    with torch.no_grad():
        # p=2 explicitly routes L_I (the inverse latent vector) to the decoder
        # Inference A: Inverse Point A + Task Param A
        out_A, _, _, _ = model(Y1_A, Y2_A, C_A, x_full, extra_pass=False, p=2, mask_indices_1=val_mask_1, mask_indices_2=val_mask_2)
        pred_inv_A = out_A.chunk(4, dim=-1)[2] # Index 2 is pred_mean_i
        
        # Inference B: Inverse Point B + Task Param A (Locked!)
        out_B, _, _, _ = model(Y1_B, Y2_B, C_A, x_full, extra_pass=False, p=2, mask_indices_1=val_mask_1, mask_indices_2=val_mask_2)
        pred_inv_B = out_B.chunk(4, dim=-1)[2]

    # 6. Un-normalize for physical analysis
    pred_A_phys = (pred_inv_A.cpu() * (Y_max_vals - Y_min_vals)) + Y_min_vals
    pred_B_phys = (pred_inv_B.cpu() * (Y_max_vals - Y_min_vals)) + Y_min_vals
    
    gt_A_phys = (Y2_A.cpu() * (Y_max_vals - Y_min_vals)) + Y_min_vals
    gt_B_phys = (Y2_B.cpu() * (Y_max_vals - Y_min_vals)) + Y_min_vals

    # Calculate model response discrepancy
    response_mse = torch.nn.functional.mse_loss(pred_A_phys, pred_B_phys).item()
    print(f"\n--- RESULTS ---")
    print(f"Mean Squared Error between Prediction A and Prediction B: {response_mse:.6f}")
    if response_mse < 0.0001:
        print("DIAGNOSIS: Posterior Collapse. The model generated the exact same curve for both. It is ignoring the spatial condition point.")
    else:
        print("DIAGNOSIS: The model is successfully responding to the t=0 spatial condition point!")

    # 7. Plotting
    pred_A_phys = pred_A_phys.squeeze(0).numpy()
    pred_B_phys = pred_B_phys.squeeze(0).numpy()
    gt_A_phys = gt_A_phys.squeeze(0).numpy()
    gt_B_phys = gt_B_phys.squeeze(0).numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Ground Truths (Inverse Trajectories)
    ax.plot(gt_A_phys[:, 0], gt_A_phys[:, 1], gt_A_phys[:, 2], color='orange', linestyle='--', alpha=0.5, label='Ground Truth Inv A')
    ax.plot(gt_B_phys[:, 0], gt_B_phys[:, 1], gt_B_phys[:, 2], color='purple', linestyle='--', alpha=0.5, label='Ground Truth Inv B')
    
    # Plot Predictions
    ax.plot(pred_A_phys[:, 0], pred_A_phys[:, 1], pred_A_phys[:, 2], color='orange', linewidth=2, label='Prediction A (Point A + Task A)')
    ax.plot(pred_B_phys[:, 0], pred_B_phys[:, 1], pred_B_phys[:, 2], color='purple', linewidth=2, label='Prediction B (Point B + Task A)')
    
    # Highlight the observed condition points (t=0)
    ax.scatter(*gt_A_phys[0], color='orange', s=100, marker='o', edgecolors='black', label='Condition Point A (t=0)')
    ax.scatter(*gt_B_phys[0], color='purple', s=100, marker='o', edgecolors='black', label='Condition Point B (t=0)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("TEMP Inverse Conditioning Isolation Test\n(Task Parameter Locked, Conditioning on t=0 of Inverse)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Update this to a TEMP pre-training run folder
    RUN_FOLDER = "model/transformer_encoded_movement_primitive/save/run_20260528_003928"
    evaluate_conditioning_response(RUN_FOLDER, dataset_path="data/paired_trajectories_insert_place")
