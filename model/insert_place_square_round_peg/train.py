import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import importlib
import model.validate_model as validate_model
import model.insert_place_square_round_peg.dual_enc_dec_model as dual_enc_dec_cnmp
import model.utils as utils
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import time

def train(model, optimizer, scheduler, EPOCHS, valid_inverses, demo_data, obs_max, d_x, d_y1, d_y2, d_param, time_len, validation_indices, training_indices, save_folder, run_id, unpaired_traj=True):

    os.makedirs(f'model/insert_place_square_round_peg/logs/run_{run_id}/', exist_ok=True)
    sys.stdout = open(f'model/insert_place_square_round_peg/logs/run_{run_id}/train_log.txt', 'w')

    training_errors = []
    validation_errors = []
    losses = []

    # Batch Size = 1 was found to be best for this dataset size
    BATCH_SIZE = 1
    d_N = len(valid_inverses)
    
    for i in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < 0.20:
                extra_pass = True

        # Note: We pass batch_size explicitly here
        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(
            extra_pass, valid_inverses, validation_indices, demo_data, 
            obs_max, d_N, d_x, d_y1, d_y2, d_param, time_len, 
            batch_size=BATCH_SIZE
        )
        
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        loss = dual_enc_dec_cnmp.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        scheduler.step()

        if i > 0 and i % 1000 == 0:
            epoch_train_error = validate_model.val_only_extra(model, training_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            training_errors.append(epoch_train_error)

            epoch_val_error = validate_model.val_only_extra(model, validation_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            validation_errors.append(epoch_val_error)
            
            losses.append(loss.item())

            # Save errors and losses
            np.save(f'{save_folder}/run_{run_id}/training_errors_mse.npy', np.array(training_errors))
            np.save(f'{save_folder}/run_{run_id}/validation_errors_mse.npy', np.array(validation_errors))
            np.save(f'{save_folder}/run_{run_id}/losses_log_prob.npy', np.array(losses))

            if min(validation_errors) == validation_errors[-1]:
                # Save model
                tqdm.write(f"Run ID: {run_id}, Saved model epoch {i}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), f'{save_folder}/run_{run_id}/best_model.pth')

    return training_errors, validation_errors, losses


if __name__ == "__main__":
    utils.seed_everything(42)

    # --- CONFIGURATION ---
    base_data_folder = "data/paired_trajectories_insert_place"
    
    # Mapping Object Name -> {Scalar ID, Paired Status}
    # 'paired': True  => Train on Forward AND Inverse
    # 'paired': False => Train on Forward ONLY (Mask Inverse)
    object_config = {
        'round_peg_4':  {'id': 0.0, 'paired': True},
        'square_peg_4': {'id': 1.0, 'paired': False} 
    }

    # Lists to hold data from ALL objects
    all_Y1_list = []
    all_Y2_list = []
    all_C_list = []
    all_valid_inverses = [] # Master list for valid_inverses

    print(f"Loading paired data from {base_data_folder}...")

    # --- DATA LOADING LOOP ---
    for obj_name, config in object_config.items():
        obj_id = config['id']
        is_paired = config['paired']
        
        obj_dir = os.path.join(base_data_folder, obj_name)
        insert_path = os.path.join(obj_dir, 'insert_all.npy')
        place_path = os.path.join(obj_dir, 'place_all.npy')

        if not os.path.exists(insert_path) or not os.path.exists(place_path):
            print(f"Warning: Could not find matched files for {obj_name} in {obj_dir}. Skipping.")
            continue
        
        print(f"  Processing {obj_name} (ID={obj_id}, Paired={is_paired})...")
        
        # Load arrays of dictionaries
        insert_data = np.load(insert_path, allow_pickle=True)
        place_data = np.load(place_path, allow_pickle=True)

        # Extract Trajectories (X, Y, Z)
        curr_Y1 = [d['pose'][0][:, :3] for d in insert_data] # Forward
        curr_Y2 = [d['pose'][0][:, :3] for d in place_data]  # Inverse

        # Limit to top X matches PER OBJECT to keep balance
        top_x_matched = min(50, len(curr_Y1))
        curr_Y1 = curr_Y1[:top_x_matched]
        curr_Y2 = curr_Y2[:top_x_matched]
        
        num_loaded = len(curr_Y1)
        print(f"    Loaded {num_loaded} trajectories.")

        if num_loaded == 0:
            continue

        # Stack into numpy arrays
        curr_Y1_np = np.stack(curr_Y1) # (N, Time, 3)
        curr_Y2_np = np.stack(curr_Y2) # (N, Time, 3)

        # Create Context for this object
        # 1. Geometric Context: (Insert_End_XY + Place_Start_XY) / 2
        insert_ends_xy = curr_Y1_np[:, -1, :2]
        place_starts_xy = curr_Y2_np[:, 0, :2]
        geom_context = (insert_ends_xy + place_starts_xy) / 2.0 # (N, 2)

        # 2. Object ID Context: Scalar value repeated for N
        id_context = np.full((num_loaded, 1), obj_id) # (N, 1)

        # 3. Combined Context: [Avg_X, Avg_Y, Obj_ID]
        curr_C_np = np.concatenate([geom_context, id_context], axis=1) # (N, 3)

        # Append to master lists
        all_Y1_list.append(curr_Y1_np)
        all_Y2_list.append(curr_Y2_np)
        all_C_list.append(curr_C_np)
        
        # Extend valid_inverses list
        # If is_paired is False, we set valid_inverses=False for these indices
        # This tells the loss function to IGNORE Y2 for these demos
        all_valid_inverses.extend([is_paired] * num_loaded)

    # --- AGGREGATE ---
    Y1 = torch.tensor(np.concatenate(all_Y1_list, axis=0), dtype=torch.float32)
    Y2 = torch.tensor(np.concatenate(all_Y2_list, axis=0), dtype=torch.float32)
    C = torch.tensor(np.concatenate(all_C_list, axis=0), dtype=torch.float32)

    # Convert valid_inverses to a simple boolean list (used by get_training_sample)
    valid_inverses = all_valid_inverses

    print(f"\nFinal Combined Data Shapes:")
    print(f"  Y1 (Forward): {Y1.shape}")
    print(f"  Y2 (Inverse): {Y2.shape}")
    print(f"  C  (Context): {C.shape}")
    print(f"  valid_inverses count: {len(valid_inverses)} (True={sum(valid_inverses)}, False={len(valid_inverses)-sum(valid_inverses)})")

    # --- NORMALIZATION (Min-Max) ---
    print("Normalizing Data (Global Min-Max)...")
    
    Y_min_vals = []
    Y_max_vals = []
    
    # Normalize Trajectories
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        
        Y_min_vals.append(min_dim)
        Y_max_vals.append(max_dim)
        
        denominator = max_dim - min_dim
        
        if denominator == 0:
            Y1[:, :, dim] = 0.0 
            Y2[:, :, dim] = 0.0
        else:
            Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / denominator
            Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / denominator

    # Normalize Context (C)
    # Note: For ID dimension (dim 2), if IDs are 0 and 1, 
    # min=0, max=1, so (val-0)/1 = val. The IDs 0.0 and 1.0 will be preserved.
    C_min_val = C.min(dim=0)[0]
    C_max_val = C.max(dim=0)[0]
    C_denom = C_max_val - C_min_val
    
    C_denom[C_denom == 0] = 1.0 
    
    C = (C - C_min_val) / C_denom
    
    print(f"Context Normalized. Range: [{C.min()}, {C.max()}]")

    # --- SETUP TRAINING VARIABLES ---
    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]

    # Create Time inputs (X)
    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

    d_x = 1
    d_param = C.shape[1] 
    d_y1 = Y1.shape[2]   
    d_y2 = Y2.shape[2]   

    OBS_MAX = 10
    d_N = num_demo

    # Split Train/Val
    all_indices = set(range(num_demo))
    validation_indices = [i for i in range(0, num_demo, 5)]
    print(f"Validation Set Size: {len(validation_indices)}")
    training_indices = list(all_indices - set(validation_indices))

    demo_data = [X1, X2, Y1, Y2, C]

    save_folder = f"model/insert_place_square_round_peg/save"
    run_id = time.time()
    os.makedirs(f'{save_folder}/run_{run_id}', exist_ok=True)

    # Save Normalization Constants for Inference
    print("Saving Normalization Stats...")
    np.save(f'{save_folder}/run_{run_id}/normalization_stats.npy', {
        'Y_min': Y_min_vals, 'Y_max': Y_max_vals,
        'C_min': C_min_val, 'C_max': C_max_val
    })

    EPOCHS = 60_001
    learning_rate = 3e-4
    
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param, dropout_p=[0.0, 0.0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

    training_errors, validation_errors, losses = train(
        model, optimizer, scheduler, EPOCHS, 
        valid_inverses, demo_data, OBS_MAX, d_x, d_y1, d_y2, d_param, time_len,
        validation_indices, training_indices, save_folder, run_id, unpaired_traj=True
    )
