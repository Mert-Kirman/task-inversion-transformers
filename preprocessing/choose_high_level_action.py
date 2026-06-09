from REASSEMBLE.io import load_h5_file
import os
import io
import shutil
import cv2
import numpy as np


def _save_video_segment(video_data, timestamps, start_time, end_time, output_path):
    """
    Extract and save a segment of video based on timestamp range.
    
    Args:
        video_data (bytes): Binary video data from H5 file.
        timestamps (np.ndarray): Array of frame timestamps.
        start_time (float): Start timestamp for the segment.
        end_time (float): End timestamp for the segment.
        output_path (str): Path where the video segment should be saved.
    """
    # Write full video to temporary file
    temp_input = "temp_full_video.mp4"
    with open(temp_input, "wb") as f:
        binary_stream = io.BytesIO(video_data)
        shutil.copyfileobj(binary_stream, f)
    
    # Find frame indices within the time range
    frame_indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    
    if len(frame_indices) == 0:
        print(f"Warning: No frames found in the time range [{start_time}, {end_time}]")
        os.remove(temp_input)
        return
    
    # Open the video file
    cap = cv2.VideoCapture(temp_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract frames
    current_frame = 0
    frame_idx_set = set(frame_indices)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in frame_idx_set:
            out.write(frame)
        
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Clean up temporary file
    os.remove(temp_input)
    print(f"Saved video segment to {output_path}")

def save_video_segment(data, start, end, video_output_dir):
    # Save the video segment
    video_keys = ['hama1', 'hama2', 'hand']
    
    for video_key in video_keys:
        video_data = data[video_key]
        video_timestamps = data['timestamps'][video_key]
        output_path = f'{video_output_dir}/{video_key}.mp4'
        _save_video_segment(video_data, video_timestamps, start, end, output_path)
        print(f"Saved {video_key} video segment")


if __name__ == '__main__':
    # Get a list of available H5 files
    h5_folder_path = f'data/original_reassemble_data/'
    available_files = [f for f in os.listdir(h5_folder_path) if f.endswith('.h5')]

    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    
    for file_name in available_files:
        h5_file_path = h5_folder_path + file_name
        data = load_h5_file(h5_file_path, decode=False)

        segments_info = data.get('segments_info', None)
        for seg_key, seg_val in segments_info.items():
            if seg_val.get('text') != b'No action.' and seg_val.get('success'):
                desired_action_word_list = seg_val.get('text').split()
                desired_action_word_list = list(map(lambda x: x.decode('utf-8').strip(". ").lower() if isinstance(x, bytes) else x.strip(". ").lower(), desired_action_word_list))

                if desired_action_word_list[0] not in ['insert', 'place']:
                    continue

                desired_action_folder_name = f'{desired_action_word_list[0]}/{"_".join(desired_action_word_list[1:])}'

                high_level_action_KEY = seg_key
                high_level_action = seg_val
                start, end = high_level_action.get('start'), high_level_action.get('end')
                
                robot_state_sensor_values = {}
                for sensor in robot_state_sensor_names:
                    indexes = np.where((data['timestamps'][sensor] >= start) & (data['timestamps'][sensor] <= end))
                    sensor_data = data['robot_state'][sensor][indexes]
                    timestamps = data['timestamps'][sensor][indexes]
                    robot_state_sensor_values[sensor] = (sensor_data, timestamps)
                    print(f"{sensor}: {sensor_data.shape}")
                
                # Save robot state sensor values
                robot_state_output_dir = f'data/raw_high_level_actions/{desired_action_folder_name}'
                os.makedirs(robot_state_output_dir, exist_ok=True)
                file_name_trimmed = file_name.replace('.h5', '')
                np.save(f'{robot_state_output_dir}/{file_name_trimmed}_{high_level_action_KEY}.npy', robot_state_sensor_values)

    # # Save video segments for a specific file
    # target_files_for_video_extraction = [
    #     "2025-01-13-19-08-03.h5",
    #     "2025-01-13-18-53-43.h5",
    #     "2025-01-10-18-31-44.h5",
    #     "2025-01-13-17-34-32.h5",
    #     "2025-01-13-16-06-01.h5",
    #     "2025-01-10-17-36-05.h5",
    #     ]
    
    # for target_file_for_video_extraction in target_files_for_video_extraction:
    #     if target_file_for_video_extraction in available_files:
    #         print(f"\nExtracting video segments from {target_file_for_video_extraction}...")
    #         h5_file_path = h5_folder_path + target_file_for_video_extraction
    #         data = load_h5_file(h5_file_path, decode=False)
    #         segments_info = data.get('segments_info', None)
    #         for seg_key, seg_val in segments_info.items():
    #             for desired_action, desired_action_folder_name in action_object_combo.items():
    #                 if seg_val.get('text') == desired_action and seg_val.get('success'):
    #                     high_level_action_KEY = seg_key
    #                     high_level_action = seg_val
    #                     start, end = high_level_action.get('start'), high_level_action.get('end')
                        
    #                     video_output_dir = f'data/videos/{target_file_for_video_extraction.replace(".h5","")}_{high_level_action_KEY}'
    #                     save_video_segment(data, start, end, video_output_dir)
