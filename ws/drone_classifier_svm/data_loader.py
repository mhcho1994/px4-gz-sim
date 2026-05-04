import os
from pathlib import Path
import numpy as np

# Import your signal processing functions
from signal_processor import (
    process_px4_flight_data, 
    process_ardu_flight_data, 
    process_real_flight_data
    # process_cogni_flight_data  # To be implemented later
)

def load_px4_dataset(folder_name, data_type='raw', measurement_type='mocap'):
    base_dir = Path("data") / folder_name
    X_timeseries = []
    y = []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_timeseries, np.array(y)

    run_folders = sorted([f for f in os.listdir(base_dir) 
                         if (f.startswith("run_") or f.startswith("run0")) and (base_dir / f).is_dir()])
    
    for run_folder in run_folders:
        if data_type == 'raw':
            px4_dir = base_dir / run_folder / "px4_logs" / "raw"
            target_ext = '.ulg'
            process_func = lambda path: process_px4_flight_data(path)
            
        elif data_type == 'processed':
            px4_dir = base_dir / run_folder / "px4_logs" / "processed"
            target_ext = '.csv'
            process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
            
        else:
            print(f"[Error] Invalid data_type: {data_type}")
            return X_timeseries, np.array(y)

        found_file = False
        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith(target_ext):
                    found_file = True
                    result = process_func(str(px4_dir / file))
                    
                    if result[0] is not None:
                        _, _, _, _, segments, _ = result
                        target_indices = [5, 7]
                        
                        count_in_run = 0
                        if segments['turn'] and len(segments['turn']) > 0:
                            for turn_segment in segments['turn']:
                                feat_turn = turn_segment['features'][:, target_indices]
                                X_timeseries.append(feat_turn) 
                                y.append(0)  # Class 0: PX4
                                count_in_run += 1
                                
                        print(f"    - {run_folder}: {count_in_run} segments extracted.")
                    else:
                        print(f"    - {run_folder}: Extraction failed (Error in processor).")
                    break
                    
        if not found_file:
            print(f"    - {run_folder}: No target file ({target_ext}) found.")

    return X_timeseries, np.array(y)

# def load_px4_dataset(folder_name, data_type='raw', measurement_type='vision'):
#     base_dir = Path("data") / folder_name
#     X_timeseries = []
#     y = []
    
#     if not base_dir.exists():
#         print(f"[Warning] Directory not found: {base_dir}")
#         return X_timeseries, np.array(y)

#     run_folders = sorted([f for f in os.listdir(base_dir) 
#                          if f.startswith("run_") and (base_dir / f).is_dir()])
    
#     print("Maximum 100 PX4 SITL data will be loaded for testing purposes.")
#     for run_folder in run_folders[:100]:  # Limit to first 100 runs for testing
#         # Route path and logic based on data_type
#         if data_type == 'raw':
#             px4_dir = base_dir / run_folder / "px4_logs" / "raw"
#             target_ext = '.ulg'
#             # 'raw' uses the SITL .ulg processing function
#             process_func = lambda path: process_px4_flight_data(path)
            
#         elif data_type == 'processed':
#             px4_dir = base_dir / run_folder / "px4_logs" / "processed"
#             target_ext = '.csv'
#             # 'processed' uses the dynamic measurement_type ('mocap' or 'vision')
#             process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
            
#         else:
#             print(f"[Error] Invalid data_type: {data_type}")
#             return X_timeseries, np.array(y)

#         if px4_dir.exists():
#             for file in os.listdir(px4_dir):
#                 if file.lower().endswith(target_ext):
#                     # Unified extraction logic
#                     result = process_func(str(px4_dir / file))
                    
#                     if result[0] is not None:
#                         # Safely unpack 6 return values
#                         _, _, _, _, segments, _ = result
#                         target_indices = [5, 7]
                        
#                         if segments['turn'] and len(segments['turn']) > 0:
#                             for turn_segment in segments['turn']:
#                                 feat_turn = turn_segment['features'][:, target_indices]
#                                 X_timeseries.append(feat_turn) 
#                                 y.append(0)  # Class 0: PX4
#                     break

#     return X_timeseries, np.array(y)

def load_ardu_dataset(folder_name, data_type='raw', measurement_type='vision'):
    base_dir = Path("data") / folder_name
    X_timeseries = []
    y = []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_timeseries, np.array(y)

    # Filtering folders starting with 'run'
    run_folders = sorted([f for f in os.listdir(base_dir) 
                         if (f.startswith("run_") or f.startswith("run0")) and (base_dir / f).is_dir()])
    
    for run_folder in run_folders:
        # Determine the target path and processing function
        if data_type == 'raw':
            ardu_dir = base_dir / run_folder / "ardu_logs" / "raw" / "logs"
            target_ext = '.bin'
            process_func = lambda path: process_ardu_flight_data(path)
        elif data_type == 'processed':
            ardu_dir = base_dir / run_folder / "ardu_logs" / "processed"
            target_ext = '.csv'
            process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
        else:
            print(f"[Error] Invalid data_type: {data_type}")
            return X_timeseries, np.array(y)

        found_file = False
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith(target_ext):
                    found_file = True
                    result = process_func(str(ardu_dir / file))
                    
                    if result[0] is not None:
                        _, _, _, _, segments, _ = result
                        target_indices = [5, 7]
                        
                        count_in_run = 0
                        if segments['turn'] and len(segments['turn']) > 0:
                            for turn_segment in segments['turn']:
                                feat_turn = turn_segment['features'][:, target_indices]
                                X_timeseries.append(feat_turn) 
                                y.append(1)  # Class 1: ArduPilot
                                count_in_run += 1
                        
                        # Output segments found for this specific run
                        print(f"    - {run_folder}: {count_in_run} segments extracted.")
                    else:
                        print(f"    - {run_folder}: Extraction failed (Error in processor).")
                    break
        
        if not found_file:
            print(f"    - {run_folder}: No target file ({target_ext}) found.")

    return X_timeseries, np.array(y)

def load_cogni_dataset(folder_name, data_type='processed', measurement_type='mocap'):
    base_dir = Path("data") / folder_name
    X_timeseries = []
    y = []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_timeseries, np.array(y)
    
    run_folders = sorted([f for f in os.listdir(base_dir) 
                         if (f.startswith("run_") or f.startswith("run0")) and (base_dir / f).is_dir()])

    for run_folder in run_folders:
        if data_type == 'processed':
            cogni_dir = base_dir / run_folder / "cogni_logs" / "processed"
            target_ext = '.csv'
            process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
        else:
            print(f"[Error] Invalid data_type: {data_type}")
            return X_timeseries, np.array(y)

        found_file = False
        if cogni_dir.exists():
            for file in os.listdir(cogni_dir):
                if file.lower().endswith(target_ext):
                    found_file = True
                    result = process_func(str(cogni_dir / file))
                    
                    if result[0] is not None:
                        _, _, _, _, segments, _ = result
                        target_indices = [5, 7]
                        
                        count_in_run = 0
                        if segments['turn'] and len(segments['turn']) > 0:
                            for turn_segment in segments['turn']:
                                feat_turn = turn_segment['features'][:, target_indices]
                                X_timeseries.append(feat_turn)
                                y.append(2)  # Class 2: Cogni
                                count_in_run += 1
                                
                        print(f"    - {run_folder}: {count_in_run} segments extracted.")
                    else:
                        print(f"    - {run_folder}: Extraction failed (Error in processor).")
                    break
                    
        if not found_file:
            print(f"    - {run_folder}: No target file ({target_ext}) found.")
                    
    return X_timeseries, np.array(y)

# def load_ardu_dataset(folder_name, data_type='raw', measurement_type='vision'):
#     base_dir = Path("data") / folder_name
#     X_timeseries = []
#     y = []
    
#     if not base_dir.exists():
#         print(f"[Warning] Directory not found: {base_dir}")
#         return X_timeseries, np.array(y)

#     run_folders = sorted([f for f in os.listdir(base_dir) 
#                          if f.startswith("run_") and (base_dir / f).is_dir()])
    
#     print("Maximum 100 ArduPilot SITL data will be loaded for testing purposes.")
#     for run_folder in run_folders[:100]:  # Limit to first 100 runs for testing
#         if data_type == 'raw':
#             ardu_dir = base_dir / run_folder / "ardu_logs" / "raw" / "logs"
#             target_ext = '.bin'
#             # 'raw' uses the SITL binary processing function
#             process_func = lambda path: process_ardu_flight_data(path)
            
#         elif data_type == 'processed':
#             ardu_dir = base_dir / run_folder / "ardu_logs" / "processed"
#             target_ext = '.csv'
#             # 'processed' uses the dynamically passed measurement_type ('mocap' or 'vision')
#             process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
            
#         else:
#             print(f"[Error] Invalid data_type: {data_type}")
#             return X_timeseries, np.array(y)

#         if ardu_dir.exists():
#             for file in os.listdir(ardu_dir):
#                 if file.lower().endswith(target_ext):
#                     result = process_func(str(ardu_dir / file))
#                     if result[0] is not None:
#                         _, _, _, _, segments, _ = result
#                         target_indices = [5, 7]
                        
#                         if segments['turn'] and len(segments['turn']) > 0:
#                             for turn_segment in segments['turn']:
#                                 feat_turn = turn_segment['features'][:, target_indices]
#                                 X_timeseries.append(feat_turn) 
#                                 y.append(1)  # Class 1: ArduPilot
#                     break

#     return X_timeseries, np.array(y)

# def load_cogni_dataset(folder_name, data_type='processed', measurement_type='vision'):
#     base_dir = Path("data") / folder_name
#     X_timeseries = []
#     y = []
    
#     if not base_dir.exists():
#         print(f"[Warning] Directory not found: {base_dir}")
#         return X_timeseries, np.array(y)
    
#     run_folders = sorted([f for f in os.listdir(base_dir) 
#                          if f.startswith("run_") and (base_dir / f).is_dir()])

#     print("Maximum 100 Cogni SITL data will be loaded for testing purposes.")
#     for run_folder in run_folders[:100]:  # Limit to first 100 runs for testing
#         # Determine the target path and processing function
#         if data_type == 'raw':
#             # 추후 Cogni raw 데이터 경로 및 함수 적용
#             # cogni_dir = base_dir / run_folder / "cogni_logs" / "raw"
#             # target_ext = '.bin' (or other extension)
#             # process_func = lambda path: process_cogni_flight_data(path)
#             print("[Info] Cogni raw data processing is not implemented yet.")
#             return X_timeseries, np.array(y)
        
#         elif data_type == 'processed':
#             cogni_dir = base_dir / run_folder / "cogni_logs" / "processed"
#             target_ext = '.csv'
#             # Pass the measurement_type to the processor
#             process_func = lambda path: process_real_flight_data(path, measurement_type=measurement_type)
#         else:
#             print(f"[Error] Invalid data_type: {data_type}")
#             return X_timeseries, np.array(y)

#         found_file = False
#         if cogni_dir.exists():
#             for file in os.listdir(cogni_dir):
#                 if file.lower().endswith(target_ext):
#                     found_file = True
#                     result = process_func(str(cogni_dir / file))
                    
#                     if result[0] is not None:
#                         _, _, _, _, segments, _ = result
#                         target_indices = [5, 7]
                        
#                         count_in_run = 0
#                         if segments['turn'] and len(segments['turn']) > 0:
#                             for turn_segment in segments['turn']:
#                                 feat_turn = turn_segment['features'][:, target_indices]
#                                 X_timeseries.append(feat_turn)
#                                 y.append(2)  # Class 2: Cogni
#                                 count_in_run += 1
                                
#                         # Output segments found for this specific run
#                         print(f"    - {run_folder}: {count_in_run} segments extracted.")
#                     else:
#                         print(f"    - {run_folder}: Extraction failed (Error in processor).")
#                     break
        
#         if not found_file:
#             print(f"    - {run_folder}: No target file ({target_ext}) found.")


#             # if cogni_dir.exists():
#             #     for file in os.listdir(cogni_dir):
#             #         if file.lower().endswith(target_ext):
#             #             result = process_func(str(cogni_dir / file))
#             #             if result[0] is not None:
#             #                 _, _, _, _, segments, _ = result
#             #                 target_indices = [5, 7]
                            
#             #                 if segments['turn'] and len(segments['turn']) > 0:
#             #                     for turn_segment in segments['turn']:
#             #                         feat_turn = turn_segment['features'][:, target_indices]
#             #                         X_timeseries.append(feat_turn)
#             #                         y.append(2)  # Class 2: Cogni
#             #             break
                    
#     return X_timeseries, np.array(y)