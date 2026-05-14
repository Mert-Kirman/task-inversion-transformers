import os
import numpy as np
import math
import matplotlib.pyplot as plt

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

def plot_example_trajectories():
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

if __name__ == "__main__":
    # plot_example_trajectories()

    # Generate 2,000 trajectories per object * 5 objects = 10,000 total pairs
    generate_synthetic_dataset(base_dir="data/synthetic_trajectories", num_objects=5, paired_samples=2000)
