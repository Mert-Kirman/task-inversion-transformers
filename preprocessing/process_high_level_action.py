import numpy as np
from scipy.interpolate import interp1d
import os

def interpolate_sensor_data(sensor_values, timestamps, target_timestamps, kind='cubic'):
    """
    Interpolate sensor data to match target timestamps using spline interpolation.
    
    Args:
        sensor_values: numpy array of shape (n_samples, n_features)
        timestamps: numpy array of shape (n_samples,) - original timestamps
        target_timestamps: numpy array of shape (m_samples,) - desired timestamps
        kind: interpolation type ('linear', 'cubic', 'quadratic', etc.)
    
    Returns:
        interpolated_values: numpy array of shape (m_samples, n_features)
    """
    # Handle 1D case (single feature)
    if sensor_values.ndim == 1:
        sensor_values = sensor_values.reshape(-1, 1)

    # --- 1. Filter out NaNs from the RAW input ---
    valid_mask = ~np.isnan(sensor_values).any(axis=1)
    clean_timestamps = timestamps[valid_mask]
    clean_values = sensor_values[valid_mask]
    
    # --- 2. Sort and remove duplicates ---
    sorted_indices = np.argsort(clean_timestamps)
    clean_timestamps = clean_timestamps[sorted_indices]
    clean_values = clean_values[sorted_indices]

    unique_timestamps, unique_indices = np.unique(clean_timestamps, return_index=True)
    clean_timestamps = clean_timestamps[unique_indices]
    clean_values = clean_values[unique_indices]

    n_features = clean_values.shape[1]
    interpolated_values = np.zeros((len(target_timestamps), n_features))
    
    # Interpolate each feature separately
    for i in range(n_features):
        # --- 3. Define Fill Values for Edges ---
        # If target_time < start_time, use the first valid value (clean_values[0, i])
        # If target_time > end_time, use the last valid value (clean_values[-1, i])
        edge_fill_values = (clean_values[0, i], clean_values[-1, i])
        
        interpolator = interp1d(clean_timestamps, clean_values[:, i], 
                               kind=kind, 
                               bounds_error=False, 
                               fill_value=edge_fill_values)
        
        interpolated_values[:, i] = interpolator(target_timestamps)
    
    return interpolated_values.squeeze()

def synchronize_multiple_modalities(modality_files, target_n_samples=100):
    """
    Synchronize multiple sensor modalities to a common number of samples.
    
    Args:
        modality_files: dict with keys as modality names and values as tuples 
                       (sensor_values, timestamps)
        target_n_samples: target number of samples for all trajectories
    
    Returns:
        synchronized_data: dict with interpolated sensor values for each modality
        common_timestamps: the common timestamp array
    """
    # Load all data
    modalities = {}
    min_time = float('inf')
    max_time = float('-inf')
    
    for modality_name, (values, timestamps) in modality_files.items():
        modalities[modality_name] = (values, timestamps)
        
        # Track time range
        min_time = min(min_time, timestamps.min())
        max_time = max(max_time, timestamps.max())
    
    # Create common timestamp array with fixed number of samples
    common_timestamps = np.linspace(min_time, max_time, target_n_samples)
    
    # Interpolate each modality
    synchronized_data = {}
    for modality_name, (values, timestamps) in modalities.items():
        print(f"Interpolating {modality_name}...")
        interpolated_sensor_data = interpolate_sensor_data(
            values, timestamps, common_timestamps, kind='linear'
        )
        synchronized_data[modality_name] = (interpolated_sensor_data, common_timestamps)
    
    return synchronized_data


if __name__ == '__main__':
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    
    # Get a list of available npy files
    raw_folder_path = 'data/raw_high_level_actions'
    available_actions = [d for d in os.listdir(raw_folder_path) if os.path.isdir(f'{raw_folder_path}/{d}')]
    for action in available_actions:
        action_path = f'{raw_folder_path}/{action}'
        objects = [o for o in os.listdir(action_path) if os.path.isdir(f'{action_path}/{o}')]
        for obj in objects:
            object_path = f'{action_path}/{obj}'
            available_files = [f for f in os.listdir(object_path) if f.endswith('.npy')]
            for file_name in available_files:
                file_path = f'{object_path}/{file_name}'
                print(f"\nProcessing {file_path}...")
                high_level_action_dict = np.load(file_path, allow_pickle=True).item()
                
                modality_files = {}
                for sensor in robot_state_sensor_names:
                    sensor_values = high_level_action_dict[sensor][0]
                    timestamps = high_level_action_dict[sensor][1]
                    modality_files[sensor] = (sensor_values, timestamps)
                    
                synchronized_data = synchronize_multiple_modalities(modality_files, target_n_samples=200)

                print("Synchronized Data Shapes:")
                for sensor in robot_state_sensor_names:
                    print(f"{sensor}: {synchronized_data[sensor][0].shape}")
                
                processed_data_dir = f'data/processed_high_level_actions/{action}/{obj}'
                print(f"\nSaving synchronized data to {processed_data_dir}...")
                os.makedirs(processed_data_dir, exist_ok=True)
                np.save(os.path.join(processed_data_dir, f'{file_name}'), synchronized_data)
