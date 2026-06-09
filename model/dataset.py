import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ReassembleDataset(Dataset):
    def __init__(self, data_dir):
        """
            X1, X2: Time vectors (N, time_len, 1)
            Y1, Y2: Forward and Inverse trajectories (N, time_len, d_y)
            C: Context parameters (N, d_param) - SPATIAL GOALS (Socket Positions) [avg_x, avg_y]
            valid_inverses: List of booleans indicating if Y2 is a paired trajectory
            time_len: Total length of the sequence
        """
        if "synthetic" not in data_dir:
            self.object_config = [
                'round_peg_1', 'round_peg_2', 'round_peg_3', 'round_peg_4',
                'small_gear', 'medium_gear', 'large_gear',
                'bnc', 'bolt_4', 'd-sub', 'ethernet', 'waterproof',
                'square_peg_1', 'square_peg_2', 'square_peg_3', 'square_peg_4', 'usb'
            ]
        else:
            self.object_config = [f'synthetic_obj_{i}' for i in range(17)]

        # Lists to hold data from ALL objects
        all_Y1_list = []
        all_Y2_list = []
        all_C_list = []
        all_valid_inverses = [] 

        print(f"Loading data from {data_dir} (Splitting Left/Right)...")

        # --- DATA LOADING LOOP ---
        for obj_name in self.object_config:
            obj_dir = os.path.join(data_dir, obj_name)
            insert_path = os.path.join(obj_dir, 'insert_all.npy')
            place_path = os.path.join(obj_dir, 'place_all.npy')

            if not os.path.exists(insert_path) or not os.path.exists(place_path):
                print(f"Warning: Could not find matched files for {obj_name} in {obj_dir}. Skipping.")
                continue
            
            print(f"  Processing {obj_name}...")
            # Load arrays of dictionaries
            insert_data = np.load(insert_path, allow_pickle=True)
            place_data = np.load(place_path, allow_pickle=True)

            # Extract Trajectories (X, Y, Z)
            curr_Y1 = [d['pose'][0][:, :3] for d in insert_data] # Forward
            curr_Y2 = [d['pose'][0][:, :3] for d in place_data]  # Inverse
            
            num_loaded = len(curr_Y1)
            print(f"    Loaded {num_loaded} trajectories.")

            if num_loaded == 0:
                continue

            # Stack into numpy arrays
            curr_Y1_np = np.stack(curr_Y1) # (N, Time, 3)
            curr_Y2_np = np.stack(curr_Y2) # (N, Time, 3)

            # Calculate Geometric Context (The Spatial Goal: (Insert_End_XY + Place_Start_XY) / 2)
            insert_ends_xy = curr_Y1_np[:, -1, :2]
            place_starts_xy = curr_Y2_np[:, 0, :2]
            geom_context = (insert_ends_xy + place_starts_xy) / 2.0 # (N, 2)

            # TASK PARAMETER 
            # Spatial Goal Configuration [avg_x, avg_y]
            curr_C_np = geom_context # (N, 2)

            # DYNAMIC SPATIAL SPLIT (Out-of-Distribution Logic)
            # Assuming origin is 0,0,0:
            # LEFT SIDE (X <= 0): Paired = True (Used for Training)
            # RIGHT SIDE (X > 0): Paired = False (Used for Zero-Shot Extrapolation)
            is_paired_array = (geom_context[:, 0] <= 0.0)

            # Append to master lists
            all_Y1_list.append(curr_Y1_np)
            all_Y2_list.append(curr_Y2_np)
            all_C_list.append(curr_C_np)
            
            # Extend valid_inverses list with our dynamically calculated booleans
            # If is_paired_array is False, we set valid_inverses=False for these indices
            # This tells the loss function to IGNORE Y2 for these demos
            all_valid_inverses.extend(is_paired_array.tolist())

        # --- AGGREGATE ---
        Y1 = torch.tensor(np.concatenate(all_Y1_list, axis=0), dtype=torch.float32)
        Y2 = torch.tensor(np.concatenate(all_Y2_list, axis=0), dtype=torch.float32)
        C = torch.tensor(np.concatenate(all_C_list, axis=0), dtype=torch.float32)
        valid_inverses = all_valid_inverses

        print(f"\nFinal Combined Data Shapes:")
        print(f"  Y1 (Forward): {Y1.shape}")
        print(f"  Y2 (Inverse): {Y2.shape}")
        print(f"  C  (Context): {C.shape}")
        print(f"  Valid Inverses (Left Side - Train): {sum(valid_inverses)}")
        print(f"  Masked Inverses (Right Side - Extrapolation Target): {len(valid_inverses)-sum(valid_inverses)}")

        num_demo = Y1.shape[0]
        time_len = Y1.shape[1]

        # Create Time inputs (X)
        X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
        X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

        self.X1 = X1
        self.X2 = X2
        self.Y1 = Y1
        self.Y2 = Y2
        self.C = C
        self.valid_inverses = valid_inverses
        
        self.d_N = num_demo
        self.time_len = time_len
        self.d_x = 1
        self.d_y1 = Y1.shape[2]
        self.d_y2 = Y2.shape[2]
        self.d_param = C.shape[1]

    def __len__(self):
        return self.d_N

    def __getitem__(self, idx):
        x1_seq = self.X1[idx]
        x2_seq = self.X2[idx]
        y1_seq = self.Y1[idx]
        y2_seq = self.Y2[idx]
        context = self.C[idx]
        is_valid_inverse = self.valid_inverses[idx]

        # Sample 1 random target point for the legacy MLP Decoders
        target_idx = torch.randint(0, self.time_len, (1,)).item()
        
        x_tar = x1_seq[target_idx].unsqueeze(0)     # Shape: (1, d_x)
        y_tar_f = y1_seq[target_idx].unsqueeze(0)   # Shape: (1, d_y1)
        y_tar_i = y2_seq[target_idx].unsqueeze(0)   # Shape: (1, d_y2)

        return {
            'y1_seq': y1_seq,       # Full forward trajectory
            'y2_seq': y2_seq,       # Full inverse trajectory
            'context': context,     # Task parameters
            'x_tar': x_tar,         # Random time point to predict
            'y_tar_f': y_tar_f,     # Ground truth forward at x_tar
            'y_tar_i': y_tar_i,     # Ground truth inverse at x_tar
            'is_valid_inverse': is_valid_inverse
        }

if __name__ == "__main__":
    dataset = ReassembleDataset("data/paired_trajectories_insert_place")
    loader = DataLoader(dataset, batch_size=16)
    batch = next(iter(loader))
    print(f'\nForward Trajectory Batch Shape: {batch["y1_seq"].shape}')
    print(f'Inverse Trajectory Batch Shape: {batch["y2_seq"].shape}')
    print(f'Context Batch Shape: {batch["context"].shape}')
    print(f'X Target Batch Shape: {batch["x_tar"].shape}')
    print(f'Forward Target Batch Shape: {batch["y_tar_f"].shape}')
    print(f'Inverse Target Batch Shape: {batch["y_tar_i"].shape}')
    print(f'Valid Inverse Batch Shape: {len(batch["is_valid_inverse"])}\n')

    dataset = ReassembleDataset("data/synthetic_trajectories")
    loader = DataLoader(dataset, batch_size=16)
    batch = next(iter(loader))
    print(f'\nForward Trajectory Batch Shape: {batch["y1_seq"].shape}')
    print(f'Inverse Trajectory Batch Shape: {batch["y2_seq"].shape}')
    print(f'Context Batch Shape: {batch["context"].shape}')
    print(f'X Target Batch Shape: {batch["x_tar"].shape}')
    print(f'Forward Target Batch Shape: {batch["y_tar_f"].shape}')
    print(f'Inverse Target Batch Shape: {batch["y_tar_i"].shape}')
    print(f'Valid Inverse Batch Shape: {len(batch["is_valid_inverse"])}')
