import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.optim as optim
import numpy as np
import model.validate_model as validate_model
import model.dual_cnmp_latent_alignment.dual_cnmp_model as dual_cnmp_model
import model.utils as utils
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from datetime import datetime


# Function for saving training configurations (epoch count, batch size, input dimensions, model name etc) in a txt file
def save_training_configs(save_folder, details_dict):
    details_path = os.path.join(save_folder, 'training_configs.txt')
    with open(details_path, 'w') as f:
        for key, value in details_dict.items():
            f.write(f"{key}: {value}\n")

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def train(model, optimizer, scheduler, EPOCHS, valid_inverses, demo_data, obs_max, d_x, d_y1, d_y2, d_param, time_len, training_indices, validation_indices, test_indices, save_folder, run_id, device, batch_size=16, unpaired_traj=True):
    sys.stdout = Logger(os.path.join(save_folder, 'train_log.txt'))

    training_errors = []
    validation_errors = []
    losses = []

    d_N = len(valid_inverses)
    
    for i in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < 0.20:
                extra_pass = True

        # Force the sampling to happen on the CPU
        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_cnmp_model.get_training_sample(
            extra_pass, valid_inverses, validation_indices, test_indices, demo_data, 
            obs_max, d_N, d_x, d_y1, d_y2, d_param, time_len, 
            batch_size=batch_size, device="cpu"
        )

        # Transfer the fully constructed tensors to the GPU all at once
        obs = obs.to(device)
        params = params.to(device)
        mask = [m.to(device) for m in mask]
        x_tar = x_tar.to(device)
        y_tar_f = y_tar_f.to(device)
        y_tar_i = y_tar_i.to(device)
        demo_data = [d.to(device) for d in demo_data]
        
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        loss = dual_cnmp_model.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        scheduler.step()

        if i > 0 and i % 400 == 0:
            epoch_train_error = validate_model.val_only_extra(model, training_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            training_errors.append(epoch_train_error if isinstance(epoch_train_error, (int, float)) else epoch_train_error.item())

            epoch_val_error = validate_model.val_only_extra(model, validation_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len, device=device)
            validation_errors.append(epoch_val_error if isinstance(epoch_val_error, (int, float)) else epoch_val_error.item())
            
            losses.append(loss.item())

            # Save errors and losses
            np.save(f'{save_folder}/training_errors_mse.npy', np.array(training_errors))
            np.save(f'{save_folder}/validation_errors_mse.npy', np.array(validation_errors))
            np.save(f'{save_folder}/losses_log_prob.npy', np.array(losses))

            if min(validation_errors) == validation_errors[-1]:
                # Save model
                tqdm.write(f"Run ID: {run_id}, Saved model epoch {i}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), f'{save_folder}/best_model.pth')

    return training_errors, validation_errors, losses


if __name__ == "__main__":
    seed = 42
    utils.seed_everything(seed)

    # --- DEVICE CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- CONFIGURATION ---
    base_data_folder = "data/paired_trajectories_insert_place"
    
    # Mapping Object Name -> {Scalar ID, Paired Status}
    # 'paired': True  => Train on Forward AND Inverse
    # 'paired': False => Train on Forward ONLY (Mask Inverse)
    object_config = {
        # ==========================================
        # PAIRED CATEGORIES (The Teachers)
        # ==========================================
        
        # Category 1: Radially Symmetric 
        'round_peg_1':  {'id': 0.0, 'paired': True,  'label': 'Round Peg 1'},
        'round_peg_2':  {'id': 1.0, 'paired': True,  'label': 'Round Peg 2'},
        'round_peg_3':  {'id': 2.0, 'paired': True,  'label': 'Round Peg 3'},
        'round_peg_4':  {'id': 3.0, 'paired': True,  'label': 'Round Peg 4'},
        
        # Category 2: Meshing / Rotational
        'small_gear':   {'id': 4.0, 'paired': True,  'label': 'Small Gear'},
        'medium_gear':  {'id': 5.0, 'paired': True,  'label': 'Medium Gear'},
        'large_gear':   {'id': 6.0, 'paired': True,  'label': 'Large Gear'},
        
        # Category 3: Asymmetric Connectors & Fasteners
        'bnc':          {'id': 7.0, 'paired': True,  'label': 'BNC Connector'},
        'bolt_4':       {'id': 8.0, 'paired': True,  'label': 'Bolt 4 / Nut'},
        'd-sub':        {'id': 9.0, 'paired': True,  'label': 'D-SUB Connector'},
        'ethernet':     {'id': 10.0, 'paired': True,  'label': 'Ethernet Connector'},
        'waterproof':   {'id': 11.0, 'paired': True,  'label': 'Waterproof Connector'},

        # ==========================================
        # UNPAIRED CATEGORIES (Zero-Shot Targets)
        # ==========================================
        
        # Zero-Shot Test 1: Corners & Edges (Highly Geometric, No Rotational Symmetry)
        'square_peg_1': {'id': 12.0, 'paired': False, 'label': 'Square Peg 1 (Unpaired)'},
        'square_peg_2': {'id': 13.0, 'paired': False, 'label': 'Square Peg 2 (Unpaired)'},
        'square_peg_3': {'id': 14.0, 'paired': False, 'label': 'Square Peg 3 (Unpaired)'},
        'square_peg_4': {'id': 15.0, 'paired': False, 'label': 'Square Peg 4 (Unpaired)'},
        
        # Zero-Shot Test 2: Highly Asymmetric Alien Shape
        'usb':          {'id': 16.0, 'paired': False, 'label': 'USB Connector (Unpaired)'}
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

    # Split Train/Val/Test (70% / 15% / 15%)
    # Split off the Test set (15% of total data)
    train_val_idx, test_indices = train_test_split(
        range(num_demo), 
        test_size=0.15, 
        stratify=valid_inverses,
        random_state=seed
    )

    # Extract labels for the remaining 85% to stratify again
    train_val_labels = [valid_inverses[i] for i in train_val_idx]

    # Split remainder into Train (70%) and Val (15%)
    training_indices, validation_indices = train_test_split(
        train_val_idx, 
        test_size=0.1764, 
        stratify=train_val_labels,
        random_state=seed
    )
    
    print(f"Split sizes -> Train: {len(training_indices)}, Val: {len(validation_indices)}, Test: {len(test_indices)}")

    # --- NORMALIZATION (Strictly on Training Data) ---
    print("Normalizing Data (Global Min-Max from Training Set)...")
    
    Y1_train = Y1[training_indices]
    Y2_train = Y2[training_indices]
    C_train = C[training_indices]

    # Combine Y1 and Y2 to find absolute physical workspace boundaries
    Y_train_combined = torch.cat([Y1_train, Y2_train], dim=0)
    
    Y_min_vals = torch.amin(Y_train_combined, dim=(0, 1))
    Y_max_vals = torch.amax(Y_train_combined, dim=(0, 1))

    C_min_val = torch.min(C_train, dim=0).values
    C_max_val = torch.max(C_train, dim=0).values

    epsilon = 1e-8
    
    # Apply to entire dataset
    Y1 = (Y1 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    Y2 = (Y2 - Y_min_vals) / (Y_max_vals - Y_min_vals + epsilon)
    C = (C - C_min_val) / (C_max_val - C_min_val + epsilon)

    # --- SAVE FOLDER SETUP ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"model/dual_cnmp_latent_alignment/save/run_{run_id}"
    os.makedirs(save_folder, exist_ok=True)

    # Save Test Indices for Evaluation Script
    np.save(os.path.join(save_folder, 'test_indices.npy'), np.array(test_indices))

    # Save Normalization Constants
    print("Saving Normalization Stats...")
    np.save(f'{save_folder}/normalization_stats.npy', {
        'Y_min': Y_min_vals.cpu(),
        'Y_max': Y_max_vals.cpu(),
        'C_min': C_min_val.cpu(),
        'C_max': C_max_val.cpu()
    })

    demo_data = [X1, X2, Y1, Y2, C]

    # --- TRAINING HYPERPARAMETERS ---
    # Increased to 18000 to match the ~18k gradient steps taken by the TEMP architecture
    EPOCHS = 18000 
    BATCH_SIZE = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout_p = [0.0, 0.0]
    
    model = dual_cnmp_model.DualCNMP(d_x, d_y1, d_y2, d_param, dropout_p=dropout_p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Save training configuration details
    training_details = {
        'model_name': 'DualCNMP',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_p': dropout_p,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'device': str(device),
        'd_x': d_x,
        'd_y1': d_y1,
        'd_y2': d_y2,
        'd_param': d_param,
        'time_len': time_len,
        'obs_max': OBS_MAX,
        'num_demonstrations': num_demo,
        'num_training_samples': len(training_indices),
        'num_validation_samples': len(validation_indices),
        'Y1_shape': Y1.shape,
        'Y2_shape': Y2.shape,
        'C_shape': C.shape,
        'objects_config': str(object_config),
        'unpaired_training': True,
        'extra_pass_probability': 0.20,
        'gradient_clip_norm': 5.0,
        'seed': 42
    }
    
    save_training_configs(save_folder, training_details)
    print(f"\nTraining configurations saved to run_{run_id}/training_configs.txt")

    training_errors, validation_errors, losses = train(
        model, optimizer, scheduler, EPOCHS, 
        valid_inverses, demo_data, OBS_MAX, d_x, d_y1, d_y2, d_param, time_len, 
        training_indices, validation_indices, test_indices, save_folder, run_id, device, 
        batch_size=BATCH_SIZE, unpaired_traj=True
    )
