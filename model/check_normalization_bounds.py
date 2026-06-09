import os
import sys
import torch
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import ReassembleDataset

def check_bounds(synthetic_stats_path):
    print("--- Sim2Real Normalization Bound Check ---\n")
    
    # Load Synthetic Bounds
    if not os.path.exists(synthetic_stats_path):
        print(f"Error: Could not find {synthetic_stats_path}")
        return
    
    checkpoint = torch.load(synthetic_stats_path)
    stats = checkpoint['norm_stats']
    Y_min_syn = stats['Y_min'].cpu()
    Y_max_syn = stats['Y_max'].cpu()
    
    print("Synthetic Y Min Bounds (X, Y, Z):", Y_min_syn.numpy())
    print("Synthetic Y Max Bounds (X, Y, Z):", Y_max_syn.numpy())
    print("-" * 50)
    
    # Load Real REASSEMBLE Data
    print("Loading Real REASSEMBLE Dataset...")
    real_dataset = ReassembleDataset("data/paired_trajectories_insert_place")
    
    # Combine forward and inverse trajectories to find absolute physical extremes
    Y_real_combined = torch.cat([real_dataset.Y1, real_dataset.Y2], dim=0)
    
    Y_min_real = torch.amin(Y_real_combined, dim=(0, 1))
    Y_max_real = torch.amax(Y_real_combined, dim=(0, 1))
    
    print("Real Data Y Min Bounds (X, Y, Z):", Y_min_real.numpy())
    print("Real Data Y Max Bounds (X, Y, Z):", Y_max_real.numpy())
    print("-" * 50)
    
    # Test: Apply Synthetic Bounds to Real Data
    epsilon = 1e-8
    Y_real_normalized = (Y_real_combined - Y_min_syn) / (Y_max_syn - Y_min_syn + epsilon)
    
    norm_min = torch.amin(Y_real_normalized, dim=(0, 1)).numpy()
    norm_max = torch.amax(Y_real_normalized, dim=(0, 1)).numpy()
    
    print("\n[RESULT] Normalized Real Data Bounds:")
    print(f"Normalized Min (Target ~0.0): {norm_min}")
    print(f"Normalized Max (Target ~1.0): {norm_max}")
    
    # Analysis
    print("\n--- DIAGNOSIS ---")
    safe = True
    for val in norm_min:
        if val < -0.15: safe = False
    for val in norm_max:
        if val > 1.15: safe = False
        
    if safe:
        print("SAFE TO FINE-TUNE. The Real dataset fits comfortably inside (or very close to) the Synthetic bounding box.")
    else:
        print("DANGER. The Real dataset significantly exceeds the Synthetic bounds. Fine-tuning may collapse. You need to regenerate the synthetic data with a wider radius/Z-height.")

if __name__ == "__main__":
    check_bounds("model/transformer_encoded_diffusion_policy/save/run_20260525_041429/best_model.pth")
