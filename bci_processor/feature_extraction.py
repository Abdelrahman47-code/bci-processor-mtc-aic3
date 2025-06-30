# bci_processor/feature_extraction.py
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, entropy
import pywt
from sklearn.cross_decomposition import CCA
import logging
from typing import List, Optional

class FeatureExtractor:
    """Extracts features for MI and SSVEP tasks."""
    
    def __init__(self, config: dict):
        self.sampling_rate = config['data']['sampling_rate']
        self.eeg_channels = config['data']['eeg_channels']
        self.mi_config = config['feature_extraction']['mi']
        self.ssvep_config = config['feature_extraction']['ssvep']
    
    def extract_wavelet_features(self, signal_data: np.ndarray, wavelet: str, levels: int) -> List[float]:
        """Extract wavelet-based features."""
        features = []
        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
        for coeff in coeffs:
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.mean(np.abs(coeff)),
                skew(coeff),
                kurtosis(coeff)
            ])
        total_energy = sum(np.sum(coeff**2) for coeff in coeffs)
        for coeff in coeffs:
            subband_energy = np.sum(coeff**2)
            features.append(subband_energy / (total_energy + 1e-10))
        return features
    
    def extract_mi_features(self, eeg_data: np.ndarray, processed_data: List[np.ndarray], label: Optional[str] = None) -> np.ndarray:
        """Extract enhanced MI features with FBCSP."""
        features = []
        motor_indices = [self.eeg_channels.index(ch) for ch in self.mi_config['motor_channels']]
        
        for band_data in processed_data:
            for ch_idx in range(band_data.shape[1]):
                channel_data = band_data[:, ch_idx]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.median(channel_data),
                    skew(channel_data),
                    kurtosis(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.mean(np.abs(np.diff(channel_data))),
                    np.std(np.diff(channel_data)),
                ])
                
                freqs, psd = signal.welch(channel_data, self.sampling_rate, nperseg=256)
                alpha_mask = (freqs >= 8) & (freqs <= 12)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                mu_mask = (freqs >= 8) & (freqs <= 13)
                features.extend([
                    np.mean(psd[alpha_mask]),
                    np.max(psd[alpha_mask]),
                    np.mean(psd[beta_mask]),
                    np.max(psd[beta_mask]),
                    np.mean(psd[mu_mask]),
                    np.sum(psd[alpha_mask]) / np.sum(psd),
                    np.sum(psd[beta_mask]) / np.sum(psd),
                ])
                psd_norm = psd / np.sum(psd)
                features.append(entropy(psd_norm))
                
                wavelet_feats = self.extract_wavelet_features(channel_data, self.mi_config['wavelet'], self.mi_config['wavelet_levels'])
                features.extend(wavelet_feats)
            
            # Simplified CSP
            cov_matrix = np.cov(band_data.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            n_filters = self.mi_config['csp_filters']
            top_filters = eigenvecs[:, -n_filters:]
            bottom_filters = eigenvecs[:, :n_filters]
            for filt in np.hstack([top_filters, bottom_filters]).T:
                filtered_signal = np.dot(band_data, filt)
                features.extend([
                    np.var(filtered_signal),
                    np.log(np.var(filtered_signal) + 1e-10),
                    np.std(np.diff(filtered_signal))
                ])
        
        # Cross-channel features
        c3_idx = self.eeg_channels.index('C3')
        c4_idx = self.eeg_channels.index('C4')
        for band_data in processed_data:
            c3_power = np.mean(band_data[:, c3_idx] ** 2)
            c4_power = np.mean(band_data[:, c4_idx] ** 2)
            features.append((c3_power - c4_power) / (c3_power + c4_power + 1e-8))
        
        return np.array(features)
    
    def extract_ssvep_features(self, eeg_data: np.ndarray, processed_data: np.ndarray) -> np.ndarray:
        """Extract enhanced SSVEP features with extended CCA."""
        features = []
        target_freqs = self.ssvep_config['target_freqs']
        harmonics = self.ssvep_config['harmonics']
        occipital_indices = [self.eeg_channels.index(ch) for ch in self.ssvep_config['occipital_channels']]
        n_fft = self.ssvep_config['n_fft']
        
        # Frequency domain features
        for ch_idx in occipital_indices:
            channel_data = processed_data[:, ch_idx]
            fft_data = fft(channel_data, n=n_fft)
            freqs = fftfreq(n_fft, 1/self.sampling_rate)
            power_spectrum = np.abs(fft_data) ** 2
            
            for target_freq in target_freqs:
                target_powers = []
                for harmonic in harmonics:
                    harmonic_freq = target_freq * harmonic
                    if harmonic_freq < self.sampling_rate / 2:
                        freq_mask = (np.abs(freqs - harmonic_freq) <= self.ssvep_config['freq_range'])
                        target_power = np.mean(power_spectrum[freq_mask])
                        target_powers.append(np.log(target_power + 1e-10))
                
                features.extend(target_powers)
                if len(target_powers) >= 2:
                    features.append(target_powers[1] - target_powers[0])
        
        # Extended CCA
        n_samples = processed_data.shape[0]
        t = np.arange(n_samples) / self.sampling_rate
        for target_freq in target_freqs:
            ref_signals = []
            for harmonic in harmonics:
                freq = target_freq * harmonic
                if freq < self.sampling_rate / 2:
                    ref_signals.extend([
                        np.sin(2 * np.pi * freq * t),
                        np.cos(2 * np.pi * freq * t),
                        np.sin(2 * np.pi * freq * t + np.pi/2)
                    ])
            ref_signals = np.array(ref_signals).T
            occipital_data = processed_data[:, occipital_indices]
            
            try:
                cca = CCA(n_components=1)
                cca.fit(occipital_data, ref_signals)
                X_c, Y_c = cca.transform(occipital_data, ref_signals)
                corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                features.extend([corr, np.max(X_c), np.std(X_c)])
            except Exception as e:
                logging.warning(f"CCA failed for freq {target_freq}: {e}")
                features.extend([0.0, 0.0, 0.0])
        
        # SNR calculation
        for target_freq in target_freqs:
            snr_values = []
            for ch_idx in occipital_indices:
                channel_data = processed_data[:, ch_idx]
                freqs_psd, psd = signal.welch(channel_data, self.sampling_rate, nperseg=1024)
                signal_mask = (freqs_psd >= target_freq - self.ssvep_config['freq_range']) & \
                             (freqs_psd <= target_freq + self.ssvep_config['freq_range'])
                noise_bands = []
                for i in range(1, 5):
                    harmonic_freq = target_freq * i
                    if harmonic_freq < self.sampling_rate / 2:
                        noise_exclude = (freqs_psd >= harmonic_freq - 0.5) & (freqs_psd <= harmonic_freq + 0.5)
                    else:
                        noise_exclude = np.zeros_like(freqs_psd, dtype=bool)
                    noise_bands.append(noise_exclude)
                
                noise_exclude_mask = np.any(noise_bands, axis=0)
                noise_mask = (freqs_psd >= 5) & (freqs_psd <= 50) & ~noise_exclude_mask
                signal_power = np.mean(psd[signal_mask])
                noise_power = np.median(psd[noise_mask])
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_values.append(snr)
            
            features.extend([
                np.mean(snr_values),
                np.max(snr_values),
                np.std(snr_values)
            ])
        
        # Phase features
        for target_freq in target_freqs:
            phase_features = []
            for ch_idx in occipital_indices:
                channel_data = processed_data[:, ch_idx]
                low_norm = max(0.01, (target_freq - 1) / (self.sampling_rate / 2))
                high_norm = min(0.99, (target_freq + 1) / (self.sampling_rate / 2))
                if low_norm < high_norm:
                    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    filtered_signal = signal.filtfilt(b, a, channel_data)
                    analytic_signal = signal.hilbert(filtered_signal)
                    instantaneous_phase = np.angle(analytic_signal)
                    plv = np.abs(np.mean(np.exp(1j * instantaneous_phase)))
                    phase_var = 1 - np.abs(np.mean(np.exp(1j * instantaneous_phase)))
                    phase_features.extend([plv, phase_var])
            
            if phase_features:
                features.extend([
                    np.mean(phase_features),
                    np.max(phase_features),
                    np.std(phase_features)
                ])
        
        return np.array(features)