import os
import numpy as np
import matplotlib.pyplot as plt
import math

def cubic_bezier(p0, p1, p2, p3, num_points=200):
    """Generates a 3D trajectory using a Cubic Bezier formula."""
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    curve = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return curve

def generate_synthetic_dataset(base_dir="data/synthetic_trajectories", num_objects=5, paired_samples=2000):
    """
    Generates synthetic robotic trajectories mimicking the REASSEMBLE dataset.
    Creates 'insert_all.npy' (Forward) and 'place_all.npy' (Inverse) for each object.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Generating Synthetic Dataset in '{base_dir}'...")
    
    for obj_id in range(num_objects):
        obj_name = f"synthetic_obj_{obj_id}"
        obj_dir = os.path.join(base_dir, obj_name)
        os.makedirs(obj_dir, exist_ok=True)
        
        insert_data = []
        place_data = []
        
        # Define base locations for this specific object in the workspace
        # Object 0 is at X=0.1, Object 1 is at X=0.2, etc.
        base_pick = np.array([0.1 + (obj_id * 0.1), 0.2, 0.0])
        base_mid  = np.array([0.1 + (obj_id * 0.1), 0.5, 0.0])
        base_drop = np.array([0.1 + (obj_id * 0.1), 0.8, 0.0])

        print(f"  Generating {paired_samples} trajectories for {obj_name}...")

        for _ in range(paired_samples):
            # --- 1. Add random Jitter to create dataset variance ---
            # Random shift between -2cm and +2cm
            pick_jitter = np.random.uniform(-0.02, 0.02, 3)
            pick_jitter[2] = 0 # Keep Z=0 for table surface
            
            mid_jitter = np.random.uniform(-0.02, 0.02, 3)
            mid_jitter[2] = 0
            
            drop_jitter = np.random.uniform(-0.02, 0.02, 3)
            drop_jitter[2] = 0
            
            pt_A = base_pick + pick_jitter
            pt_B = base_mid + mid_jitter
            pt_C = base_drop + drop_jitter
            
            # --- 2. Generate FORWARD Trajectory (pt_A to pt_B) ---
            # Control points lift up in Z (0.15) to simulate picking up
            fwd_p1 = pt_A + np.array([0, 0.05, 0.15]) 
            fwd_p2 = pt_B + np.array([0, -0.05, 0.15])
            fwd_traj = cubic_bezier(pt_A, fwd_p1, fwd_p2, pt_B, 200)
            
            # --- 3. Generate INVERSE Trajectory (pt_B to pt_C) ---
            inv_p1 = pt_B + np.array([0, 0.05, 0.15])
            inv_p2 = pt_C + np.array([0, -0.05, 0.15])
            inv_traj = cubic_bezier(pt_B, inv_p1, inv_p2, pt_C, 200)
            
            # --- 4. Format exactly like REASSEMBLE dataset ---
            # dataset.py expects: [d['pose'][0][:, :3] for d in data]
            insert_data.append({'pose': [fwd_traj]})
            place_data.append({'pose': [inv_traj]})
            
        # Save to disk
        np.save(os.path.join(obj_dir, 'insert_all.npy'), insert_data)
        np.save(os.path.join(obj_dir, 'place_all.npy'), place_data)
        
    print("\nGeneration Complete!")
    print("To use this, update dataset.py object_config to point to 'synthetic_obj_0' through 'synthetic_obj_4'.")

def plot_example_trajectories_single_object():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Base coordinates for Object 0
    base_pick = np.array([0.1, 0.2, 0.0])
    base_mid  = np.array([0.1, 0.5, 0.0])
    base_drop = np.array([0.1, 0.8, 0.0])

    # Plot 50 samples
    for _ in range(50):
        # Jitter X and Y, keep Z=0 (table level)
        pick_jitter = np.random.uniform(-0.02, 0.02, 3); pick_jitter[2] = 0
        mid_jitter = np.random.uniform(-0.02, 0.02, 3); mid_jitter[2] = 0
        drop_jitter = np.random.uniform(-0.02, 0.02, 3); drop_jitter[2] = 0
        
        pt_A = base_pick + pick_jitter
        pt_B = base_mid + mid_jitter
        pt_C = base_drop + drop_jitter
        
        # Generate curves with an apex of Z=0.15 (15cm high)
        fwd_p1 = pt_A + np.array([0, 0.05, 0.15]) 
        fwd_p2 = pt_B + np.array([0, -0.05, 0.15])
        fwd_traj = cubic_bezier(pt_A, fwd_p1, fwd_p2, pt_B, 200)
        
        inv_p1 = pt_B + np.array([0, 0.05, 0.15])
        inv_p2 = pt_C + np.array([0, -0.05, 0.15])
        inv_traj = cubic_bezier(pt_B, inv_p1, inv_p2, pt_C, 200)
        
        ax.plot(fwd_traj[:, 0], fwd_traj[:, 1], fwd_traj[:, 2], color='blue', alpha=0.2)
        ax.plot(inv_traj[:, 0], inv_traj[:, 1], inv_traj[:, 2], color='orange', alpha=0.2)

    ax.scatter(*base_pick, color='red', s=100, label='Pick Zone')
    ax.scatter(*base_mid, color='green', s=100, label='Mid Zone')
    ax.scatter(*base_drop, color='purple', s=100, label='Drop Zone')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Synthetic Pick-and-Place (Forward=Blue, Inverse=Orange)')
    plt.legend()
    plt.show()

def plot_example_trajectories_multiple_objects():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for obj_id in range(5):
        # Define base locations for this specific object in the workspace
        # Object 0 is at X=0.1, Object 1 is at X=0.2, etc.
        base_pick = np.array([0.1 + (obj_id * 0.1), 0.2, 0.0])
        base_mid  = np.array([0.1 + (obj_id * 0.1), 0.5, 0.0])
        base_drop = np.array([0.1 + (obj_id * 0.1), 0.8, 0.0])

        # Plot 20 samples
        for _ in range(20):
            # Jitter X and Y, keep Z=0 (table level)
            pick_jitter = np.random.uniform(-0.02, 0.02, 3); pick_jitter[2] = 0
            mid_jitter = np.random.uniform(-0.02, 0.02, 3); mid_jitter[2] = 0
            drop_jitter = np.random.uniform(-0.02, 0.02, 3); drop_jitter[2] = 0
            
            pt_A = base_pick + pick_jitter
            pt_B = base_mid + mid_jitter
            pt_C = base_drop + drop_jitter
            
            # Generate curves with an apex of Z=0.15 (15cm high)
            fwd_p1 = pt_A + np.array([0, 0.05, 0.15]) 
            fwd_p2 = pt_B + np.array([0, -0.05, 0.15])
            fwd_traj = cubic_bezier(pt_A, fwd_p1, fwd_p2, pt_B, 200)
            
            inv_p1 = pt_B + np.array([0, 0.05, 0.15])
            inv_p2 = pt_C + np.array([0, -0.05, 0.15])
            inv_traj = cubic_bezier(pt_B, inv_p1, inv_p2, pt_C, 200)
            
            ax.plot(fwd_traj[:, 0], fwd_traj[:, 1], fwd_traj[:, 2], color='blue', alpha=0.2)
            ax.plot(inv_traj[:, 0], inv_traj[:, 1], inv_traj[:, 2], color='orange', alpha=0.2)

        ax.scatter(*base_pick, color='red', s=100, label='Pick Zone' if obj_id == 0 else "")
        ax.scatter(*base_mid, color='green', s=100, label='Mid Zone' if obj_id == 0 else "")
        ax.scatter(*base_drop, color='purple', s=100, label='Drop Zone' if obj_id == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Synthetic Pick-and-Place (Forward=Blue, Inverse=Orange)')
    plt.legend()
    plt.show()

def plot_reassemble_trajectories(num_objects=5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    data_dir = "data/paired_trajectories_insert_place"
    objects = os.listdir(data_dir)
    objects = objects[:num_objects] # Limit to specified number of objects for clarity
    print(f"Plotting trajectories for objects: {objects}")

    for obj in objects:
        obj_dir = os.path.join(data_dir, obj)
        insert_path = os.path.join(obj_dir, 'insert_all.npy')
        place_path = os.path.join(obj_dir, 'place_all.npy')

        if not os.path.exists(insert_path) or not os.path.exists(place_path):
            print(f"Warning: Could not find matched files for {obj} in {obj_dir}. Skipping.")
            continue

        insert_trajs = np.load(insert_path, allow_pickle=True)
        place_trajs = np.load(place_path, allow_pickle=True)

        insert_trajs = [d['pose'][0][:, :3] for d in insert_trajs]
        place_trajs = [d['pose'][0][:, :3] for d in place_trajs]

        # Plot 20 samples
        for i in range(20):
            fwd_traj = insert_trajs[i]
            inv_traj = place_trajs[i]

            ax.plot(fwd_traj[:, 0], fwd_traj[:, 1], fwd_traj[:, 2], color='blue', alpha=0.2)
            ax.plot(inv_traj[:, 0], inv_traj[:, 1], inv_traj[:, 2], color='orange', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reassembled Insert-and-Place Trajectories (Forward=Blue, Inverse=Orange)')
    plt.legend()
    plt.show()

def generate_reassemble_synthetic_dataset(base_dir="data/synthetic_trajectories", num_objects=5, paired_samples=2000, plot=False):
    '''
    Generates a synthetic dataset where all trajectories are relativized to the origin (0,0,0) (Similar to REASSEMBLE Dataset).
    '''
    if plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        os.makedirs(base_dir, exist_ok=True)
    
    print(f"Generating Relativized Synthetic Dataset in '{base_dir}'...")
    
    # The standardized Pick/Drop area is the origin
    base_hub = np.array([0.0, 0.0, 0.0])

    for obj_id in range(num_objects):
        obj_name = f"synthetic_obj_{obj_id}"
        if not plot:
            obj_dir = os.path.join(base_dir, obj_name)
            os.makedirs(obj_dir, exist_ok=True)
            
            insert_data = []
            place_data = []
        
        # Differentiator: The "Insertion Wiggle Signature" at the Spoke
        wiggle_freq = 10 + (obj_id * 15)  
        wiggle_amp = 0.005 + (obj_id * 0.002) 

        for _ in range(paired_samples):
            # Generate Spoke (Insertion Point)
            angle = np.random.uniform(0, 2 * math.pi)
            radius = np.random.uniform(0.1, 0.6)
            spoke_z_jitter = np.random.uniform(-0.03, 0.03) 
            pt_spoke = np.array([radius * math.cos(angle), radius * math.sin(angle), spoke_z_jitter])
            
            # Hub (Origin) has slight jitter
            pt_hub_pick = base_hub + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0])
            pt_hub_drop = base_hub + np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0])

            # Hungarian Matching Gap
            spoke_gap = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), np.random.uniform(0.01, 0.05)])
            pt_spoke_fwd_end = pt_spoke 
            pt_spoke_inv_start = pt_spoke + spoke_gap

            # ==========================================
            # MESSY REALITY INJECTIONS
            # ==========================================
            
            # 1. Blurry Z-Shells (Vary the arc height for every trajectory)
            base_z_clearance = 0.15
            noisy_z_clearance = base_z_clearance + np.random.normal(0, 0.02)
            
            # 2. Flattened Arcs (Pull control points horizontally towards the middle)
            flatten_factor = 0.35 
            
            # Forward (Pick -> Insert)
            fwd_p1 = pt_hub_pick + (pt_spoke_fwd_end - pt_hub_pick) * flatten_factor + np.array([0, 0, noisy_z_clearance])
            fwd_p2 = pt_spoke_fwd_end - (pt_spoke_fwd_end - pt_hub_pick) * flatten_factor + np.array([0, 0, noisy_z_clearance])
            fwd_traj = cubic_bezier(pt_hub_pick, fwd_p1, fwd_p2, pt_spoke_fwd_end, 200)
            
            # Inverse (Extract -> Drop)
            # Pull inv_p1 up slightly higher to clear the hole before flattening out
            inv_p1 = pt_spoke_inv_start - (pt_spoke_inv_start - pt_hub_drop) * (flatten_factor * 0.5) + np.array([0, 0, noisy_z_clearance - 0.06])
            inv_p2 = pt_hub_drop + (pt_spoke_inv_start - pt_hub_drop) * flatten_factor + np.array([0, 0, noisy_z_clearance])
            inv_traj = cubic_bezier(pt_spoke_inv_start, inv_p1, inv_p2, pt_hub_drop, 200)

            # 3. Mid-Flight XY Wandering
            t = np.linspace(0, 1, 200)[:, np.newaxis]
            wander_mask = np.sin(t * math.pi) # Sine wave peaks at t=0.5
            
            wander_x = np.random.uniform(-0.04, 0.04)
            wander_y = np.random.uniform(-0.04, 0.04)
            
            fwd_traj[:, 0] += (wander_mask * wander_x).squeeze()
            fwd_traj[:, 1] += (wander_mask * wander_y).squeeze()
            
            # Wander back in a slightly different path
            inv_traj[:, 0] += (wander_mask * -wander_x * 0.8).squeeze()
            inv_traj[:, 1] += (wander_mask * -wander_y * 0.8).squeeze()

            # ==========================================
            # END MESSY REALITY INJECTIONS
            # ==========================================

            # INJECT INSERTION WIGGLES (Last 20% of Forward Trajectory, approaching Spoke)
            fwd_wiggle_mask = np.where(t > 0.8, (t - 0.8) * 5, 0)
            fwd_traj[:, 0] += (np.sin(t * wiggle_freq) * wiggle_amp * fwd_wiggle_mask).squeeze()
            fwd_traj[:, 1] += (np.cos(t * wiggle_freq * 1.2) * wiggle_amp * fwd_wiggle_mask).squeeze()
            
            # Inverse extraction wiggles (First 10% of Inverse Trajectory, pulling out of Spoke)
            inv_wiggle_mask = np.where(t < 0.1, (0.1 - t) * 10, 0)
            inv_traj[:, 0] += (np.sin(t * wiggle_freq) * wiggle_amp * inv_wiggle_mask).squeeze()
            inv_traj[:, 1] += (np.cos(t * wiggle_freq * 1.2) * wiggle_amp * inv_wiggle_mask).squeeze()

            if plot:
                ax.plot(fwd_traj[:, 0], fwd_traj[:, 1], fwd_traj[:, 2], color='blue', alpha=0.2)
                ax.plot(inv_traj[:, 0], inv_traj[:, 1], inv_traj[:, 2], color='orange', alpha=0.2)
            else:
                insert_data.append({'pose': [fwd_traj]})
                place_data.append({'pose': [inv_traj]})
            
        if not plot:
            np.save(os.path.join(obj_dir, 'insert_all.npy'), insert_data)
            np.save(os.path.join(obj_dir, 'place_all.npy'), place_data)
        print(f"  Generated {paired_samples} for {obj_name}. Wiggle Freq: {wiggle_freq}.")

    if plot:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Relativized Synthetic Trajectories (Forward=Blue, Inverse=Orange)')
        plt.legend()
        plt.show()
    
    print("\nGeneration Complete!")

if __name__ == "__main__":
    # plot_example_trajectories_single_object()
    # plot_example_trajectories_multiple_objects()
    # plot_reassemble_trajectories(num_objects=5)
    generate_reassemble_synthetic_dataset(num_objects=5, paired_samples=20, plot=True)
