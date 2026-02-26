from matplotlib import pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    timestamps_interpolated =  np.linspace(0, 1, 200)
    
    processed_folder_path = 'data/processed_high_level_actions'
    available_actions = [d for d in os.listdir(processed_folder_path) if os.path.isdir(f'{processed_folder_path}/{d}')]
    for action in available_actions:
        action_path = f'{processed_folder_path}/{action}'
        objects = [o for o in os.listdir(action_path) if os.path.isdir(f'{action_path}/{o}')]
        for obj in objects:
            object_path = f'{action_path}/{obj}'
            available_files = [f for f in os.listdir(object_path) if f.endswith('.npy')]
            action_object_pairs = list()
            for file_name in available_files:
                # Get interpolated data
                file_path_interpolated = f'{processed_folder_path}/{action}/{obj}/{file_name}'
                if not os.path.exists(file_path_interpolated):
                    print(f"Interpolated file not found: {file_path_interpolated}, skipping...")
                    continue
                high_level_action_dict_interpolated = np.load(file_path_interpolated, allow_pickle=True).item()
                
                modality_files_interpolated = {}
                for sensor in robot_state_sensor_names:
                    sensor_values = high_level_action_dict_interpolated[sensor][0]
                    timestamps = high_level_action_dict_interpolated[sensor][1]
                    modality_files_interpolated[sensor] = (sensor_values, timestamps)

                action_object_pairs.append((file_name, modality_files_interpolated))

            # Plot comparisons
            sensors_to_plot = ['pose']
            rows, cols = 2, 4
            data_plots_dir = f'data/plots/{action}/{obj}'
            os.makedirs(data_plots_dir, exist_ok=True)
            for sensor in sensors_to_plot:
                plt.figure(figsize=(20, 15))
                for dim in range(action_object_pairs[0][1][sensor][0].shape[1]):
                    modality_name = f"{sensor}_{dim}"
                    plot_count = 0

                    plt.subplot(rows, cols, dim + 1)
                    for file_name, modality_files_interpolated in action_object_pairs:
                        # Interpolated data
                        sensor_values_interpolated, _ = modality_files_interpolated[sensor]
                        sensor_values_interpolated_dim = sensor_values_interpolated[:, dim]

                        if np.isnan(sensor_values_interpolated_dim).any():
                            print(f"There are NaN values in {modality_name}, skipping plot.")
                            print(sensor_values_interpolated_dim)
                            continue
                        
                        # Plot multiple interpolated values for the same dimension
                        plt.plot(timestamps_interpolated, sensor_values_interpolated_dim, label=f'{file_name}')
                        # plot_count += 1
                        # if plot_count == 3:
                        #     break  # Limit to first 3 plots for clarity
                    plt.title(f'Interpolated {modality_name} Sensor Data')
                    plt.xlabel('Time')
                    plt.ylabel('Sensor Values')
                    # plt.legend()
                    plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(data_plots_dir, f'interpolated_{sensor}_all_dims_without_dtw.png'))
                plt.close()
