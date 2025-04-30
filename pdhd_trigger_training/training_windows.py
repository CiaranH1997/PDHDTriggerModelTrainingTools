import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Load and preprocess binned time series data
def load_data(time_binned_adcsum_data, normalise=True):
    data = []
    
    for event_id in time_binned_adcsum_data:
        for apa in time_binned_adcsum_data[event_id]:
            for sub_event in time_binned_adcsum_data[event_id][apa]:
                data.append(np.array(sub_event))
    
    data_np = np.stack(data)  # Stack into 3D array
    print(f'data_np shape = {data_np.shape}')
    min_val, max_val = np.min(data_np), np.max(data_np)
    print('calculated max value: ', max_val)
    print('calculated min value: ', min_val)
    scaled_data = data_np
    # Normalize the data
    if (normalise):
        scaled_data = (data_np - min_val) / (max_val - min_val)
    
    # Split into training and testing sets
    split_idx = int(0.8 * len(scaled_data))
    train_data, test_data = scaled_data[:split_idx], scaled_data[split_idx:]
    
    return train_data, test_data, min_val, max_val#, scaler

# Load and preprocess binned time series data
def load_neutrino_data(time_binned_adcsum_data, normalise=True, min_train=0, max_train=1):
    data = []
    for event_id in time_binned_adcsum_data:
        for apa in time_binned_adcsum_data[event_id]:
            for sub_event in time_binned_adcsum_data[event_id][apa]:
                data.append(np.array(sub_event))

    data_np = np.stack(data)  # Stack into 3D array
    print(f'data_np shape = {data_np.shape}')
 
    scaled_data = data_np
    if (normalise):
        scaled_data = (data_np - min_train) / (max_train - min_train)
    
    return scaled_data