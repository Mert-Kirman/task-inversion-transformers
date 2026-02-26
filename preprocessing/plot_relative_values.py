from matplotlib import pyplot as plt
import numpy as np
import os

# --- Helper Functions for Quaternion Math ---
def quaternion_inverse(q):
    """Computes the inverse (conjugate) of a quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions q1 * q2."""
    if q1.ndim == 1: q1 = q1[np.newaxis, :]
    if q2.ndim == 1: q2 = q2[np.newaxis, :]
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack((w, x, y, z), axis=1)

if __name__ == "__main__":
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    timestamps_interpolated =  np.linspace(0, 1, 200)
    
    # Path to where the folders 'insert', 'place', etc. are located
    processed_folder_path = 'data/processed_high_level_actions'
    
    # Filter for relevant actions only
    target_actions = ['insert', 'place']
    available_actions = [d for d in os.listdir(processed_folder_path) 
                         if os.path.isdir(os.path.join(processed_folder_path, d)) and d in target_actions]
    
    if not available_actions:
        print(f"No actions found matching {target_actions} in {processed_folder_path}")

    for action in available_actions:
        action_path = os.path.join(processed_folder_path, action)
        objects = [o for o in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, o))]
        
        for obj in objects:
            object_path = os.path.join(action_path, obj)
            available_files = [f for f in os.listdir(object_path) if f.endswith('.npy')]
            
            action_object_pairs = list()
            
            print(f"Processing {action}/{obj}...")

            for file_name in available_files:
                file_path_interpolated = os.path.join(object_path, file_name)
                
                try:
                    high_level_action_dict_interpolated = np.load(file_path_interpolated, allow_pickle=True).item()
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    continue
                
                modality_files_interpolated = {}
                
                # --- LOAD POSE ---
                pose_data = high_level_action_dict_interpolated['pose'][0] # [x, y, z, qw, qx, qy, qz]
                pose_timestamps = high_level_action_dict_interpolated['pose'][1]
                num_points = pose_data.shape[0]

                # --- DETERMINE REFERENCE POINT ---
                ref_pos = None
                ref_quat = None

                if action == 'insert':
                    # Logic: Reference is START (t=0)
                    # This sets the "Table Position" (where the robot starts) as (0,0,0)
                    ref_idx = 0
                    ref_pos = pose_data[ref_idx, :3]
                    ref_quat = pose_data[ref_idx, 3:]

                elif action == 'place':
                    # Logic: Reference is MIN Z in 2nd Half
                    # This sets the "Table Position" (where the robot places) as (0,0,0)
                    search_start_idx = int(num_points * 0.50) # Look in second half
                    zs = pose_data[:, 2]
                    local_min_idx = np.argmin(zs[search_start_idx:])
                    ref_idx = search_start_idx + local_min_idx
                    
                    ref_pos = pose_data[ref_idx, :3]
                    ref_quat = pose_data[ref_idx, 3:]
                
                else:
                    print(f"Skipping undefined action logic for: {action}")
                    continue

                # --- CALCULATE RELATIVE POSE ---
                
                # A. Relative Position
                # Subtract reference position from all points
                relative_pos = pose_data[:, :3] - ref_pos
                
                # B. Relative Orientation
                # Q_relative = Q_current * Q_ref_inverse
                quaternions = pose_data[:, 3:]
                ref_quat_inv = quaternion_inverse(ref_quat)
                relative_quat = quaternion_multiply(quaternions, ref_quat_inv)
                
                # Combine back
                relative_pose = np.hstack((relative_pos, relative_quat))
                
                # Store back
                modality_files_interpolated['pose'] = (relative_pose, pose_timestamps)
                action_object_pairs.append((file_name, modality_files_interpolated))

            # --- PLOTTING ---
            sensors_to_plot = ['pose']
            rows, cols = 2, 4
            data_plots_dir = f'data/plots_relative_full/{action}/{obj}'
            os.makedirs(data_plots_dir, exist_ok=True)
            
            for sensor in sensors_to_plot:
                plt.figure(figsize=(20, 15))
                if not action_object_pairs: continue
                num_dims = action_object_pairs[0][1][sensor][0].shape[1]

                for dim in range(num_dims):
                    modality_name = f"{sensor}_{dim}"
                    dim_label = ""
                    if sensor == 'pose':
                        if dim == 0: dim_label = " (Rel X)"
                        elif dim == 1: dim_label = " (Rel Y)"
                        elif dim == 2: dim_label = " (Rel Z)"
                        elif dim == 3: dim_label = " (Rel Qw - Identity=1)"
                        else: dim_label = f" (Rel Q{dim-3} - Identity=0)"

                    plt.subplot(rows, cols, dim + 1)
                    plot_count = 0
                    for file_name, modality_files in action_object_pairs:
                        sensor_values, _ = modality_files[sensor]
                        sensor_values_dim = sensor_values[:, dim]
                        if np.isnan(sensor_values_dim).any(): continue
                        plt.plot(timestamps_interpolated, sensor_values_dim, label=f'{file_name}', alpha=0.7)
                        plot_count += 1
                        if plot_count == 55: break 
                    
                    plt.title(f'{modality_name}{dim_label}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.grid(True, alpha=0.3)
                    # if dim == 0: plt.legend(fontsize='x-small')
                
                plt.suptitle(f"Relative Pose (Ref: {'Start' if action=='insert' else 'Table Contact'})\n({action} - {obj})", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(data_plots_dir, f'relative_{sensor}_full.png'))
                plt.close()
                print(f"Saved plot to {data_plots_dir}")

            # --- SAVE PROCESSED FILES ---
            processed_save_dir = f'data/processed_relative_high_level_actions/{action}/{obj}'
            os.makedirs(processed_save_dir, exist_ok=True)
            for file_name, modality_files in action_object_pairs:
                save_path = os.path.join(processed_save_dir, file_name)
                np.save(save_path, modality_files)
            print(f"Saved processed files to {processed_save_dir}")
