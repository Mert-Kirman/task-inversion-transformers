import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ReassembleDataset(Dataset):
    def __init__(self, data_dir):
        """
            X1, X2: Time vectors (N, time_len, 1)
            Y1, Y2: Forward and Inverse trajectories (N, time_len, d_y)
            C: Context parameters (N, d_param)
            valid_inverses: List of booleans indicating if Y2 is a paired trajectory
            time_len: Total length of the sequence
        """
        # Mapping Object Name -> {Scalar ID, Paired Status}
        # 'paired': True  => Train on Forward AND Inverse
        # 'paired': False => Train on Forward ONLY (Mask Inverse)
        if "synthetic" not in data_dir:
            self.object_config = {
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
        else:
            self.object_config = {
                'synthetic_obj_0': {'id': 0.0, 'paired': True,  'label': 'Synthetic Object 0'},
                'synthetic_obj_1': {'id': 1.0, 'paired': True,  'label': 'Synthetic Object 1'},
                'synthetic_obj_2': {'id': 2.0, 'paired': True,  'label': 'Synthetic Object 2'},
                'synthetic_obj_3': {'id': 3.0, 'paired': True,  'label': 'Synthetic Object 3'},
                'synthetic_obj_4': {'id': 4.0, 'paired': True,  'label': 'Synthetic Object 4'},
                'synthetic_obj_5': {'id': 5.0, 'paired': True,  'label': 'Synthetic Object 5'},
                'synthetic_obj_6': {'id': 6.0, 'paired': True,  'label': 'Synthetic Object 6'},
                'synthetic_obj_7': {'id': 7.0, 'paired': True,  'label': 'Synthetic Object 7'},
                'synthetic_obj_8': {'id': 8.0, 'paired': True,  'label': 'Synthetic Object 8'},
                'synthetic_obj_9': {'id': 9.0, 'paired': False,  'label': 'Synthetic Object 9 (Unpaired)'},
                'synthetic_obj_10': {'id': 10.0, 'paired': False,  'label': 'Synthetic Object 10 (Unpaired)'},
                'synthetic_obj_11': {'id': 11.0, 'paired': False,  'label': 'Synthetic Object 11 (Unpaired)'},
                'synthetic_obj_12': {'id': 12.0, 'paired': False,  'label': 'Synthetic Object 12 (Unpaired)'},
                'synthetic_obj_13': {'id': 13.0, 'paired': True,  'label': 'Synthetic Object 13'},
                'synthetic_obj_14': {'id': 14.0, 'paired': True,  'label': 'Synthetic Object 14'},
                'synthetic_obj_15': {'id': 15.0, 'paired': False,  'label': 'Synthetic Object 15 (Unpaired)'},
                'synthetic_obj_16': {'id': 16.0, 'paired': True,  'label': 'Synthetic Object 16'},
            }

        # Lists to hold data from ALL objects
        all_Y1_list = []
        all_Y2_list = []
        all_C_list = []
        all_valid_inverses = [] # Master list for valid_inverses

        print(f"Loading paired data from {data_dir}...")

        # --- DATA LOADING LOOP ---
        for obj_name, config in self.object_config.items():
            obj_id = config['id']
            is_paired = config['paired']
            
            obj_dir = os.path.join(data_dir, obj_name)
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
        valid_inverses = all_valid_inverses

        print(f"\nFinal Combined Data Shapes:")
        print(f"  Y1 (Forward): {Y1.shape}")
        print(f"  Y2 (Inverse): {Y2.shape}")
        print(f"  C  (Context): {C.shape}")
        print(f"  valid_inverses count: {len(valid_inverses)} (True={sum(valid_inverses)}, False={len(valid_inverses)-sum(valid_inverses)})")

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
        # Grab the full sequences for the BERT Encoders
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
