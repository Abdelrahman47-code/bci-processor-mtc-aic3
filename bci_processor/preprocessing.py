# bci_processor/preprocessing.py
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
import logging
from typing import Union, List, Tuple
from pathlib import Path

class Preprocessor:
    """Handles EEG data loading and preprocessing."""
    
    def __init__(self, config: dict):
        self.base_path = Path(config['data']['base_path'])
        self.sampling_rate = config['data']['sampling_rate']
        self.eeg_channels = config['data']['eeg_channels']
        self.mi_samples_per_trial = config['data']['mi_samples_per_trial']
        self.ssvep_samples_per_trial = config['data']['ssvep_samples_per_trial']
        self.mi_freq_bands = config['preprocessing']['mi_freq_bands']
        self.ssvep_freq_band = config['preprocessing']['ssvep_freq_band']
        self.notch_freq = config['preprocessing']['notch_freq']
        self.notch_quality = config['preprocessing']['notch_quality']
        self.butter_order = config['preprocessing']['butter_order']
        self.ica_components = config['preprocessing']['ica_components']
    
    def load_trial_data(self, row: pd.Series) -> np.ndarray:
        """Load EEG data for a specific trial with error handling."""
        id_num = row['id']
        dataset = 'train' if id_num <= 4800 else 'validation' if id_num <= 4900 else 'test'
        eeg_path = self.base_path / row['task'] / dataset / row['subject_id'] / str(row['trial_session']) / 'EEGdata.csv'
        try:
            eeg_data = pd.read_csv(eeg_path)
        except Exception as e:
            logging.error(f"Error loading {eeg_path}: {e}")
            raise
        
        trial_num = int(row['trial'])
        samples_per_trial = self.mi_samples_per_trial if row['task'] == 'MI' else self.ssvep_samples_per_trial
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial
        
        if end_idx > len(eeg_data):
            end_idx = len(eeg_data)
            start_idx = max(0, end_idx - samples_per_trial)
        
        trial_data = eeg_data.iloc[start_idx:end_idx][self.eeg_channels]
        trial_array = trial_data.values.astype(np.float32)
        
        # Subject-specific normalization
        trial_array = (trial_array - np.mean(trial_array, axis=0)) / (np.std(trial_array, axis=0) + 1e-10)
        
        # NaN handling with interpolation
        if np.isnan(trial_array).any():
            for ch in range(trial_array.shape[1]):
                channel_data = trial_array[:, ch]
                nan_mask = np.isnan(channel_data)
                if nan_mask.any():
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) > 1:
                        trial_array[nan_mask, ch] = np.interp(
                            np.where(nan_mask)[0], valid_indices, channel_data[valid_indices]
                        )
                    else:
                        trial_array[:, ch] = np.nanmean(channel_data)
        
        return trial_array
    
    def preprocess(self, data: np.ndarray, task_type: str = 'MI') -> Union[np.ndarray, List[np.ndarray]]:
        """Apply preprocessing steps including ICA, notch, and bandpass filtering."""
        # Scale data
        data_scaled = data / 1000.0
        data_scaled = data_scaled - np.mean(data_scaled, axis=0)
        
        # ICA for artifact removal
        try:
            ica = FastICA(n_components=self.ica_components, random_state=42)
            ica_components = ica.fit_transform(data_scaled)
            ica_components[:, 0] = 0  # Zero out first component (assumed artifact)
            data_cleaned = ica.inverse_transform(ica_components)
        except Exception as e:
            logging.warning(f"ICA failed: {e}, proceeding without ICA")
            data_cleaned = data_scaled
        
        # Notch filter
        nyquist = self.sampling_rate / 2
        b_notch, a_notch = signal.iirnotch(self.notch_freq, self.notch_quality, self.sampling_rate)
        data_notched = signal.filtfilt(b_notch, a_notch, data_cleaned, axis=0)
        
        # Task-specific bandpass filtering
        bands = self.mi_freq_bands if task_type == 'MI' else [self.ssvep_freq_band]
        filtered_data = []
        for low_freq, high_freq in bands:
            low_norm = max(0.01, min(low_freq / nyquist, 0.99))
            high_norm = max(low_norm + 0.01, min(high_freq / nyquist, 0.99))
            b, a = signal.butter(self.butter_order, [low_norm, high_norm], btype='band')
            band_data = signal.filtfilt(b, a, data_notched, axis=0)
            filtered_data.append(band_data)
        
        return filtered_data if task_type == 'MI' else filtered_data[0]