# tests/test_feature_extraction.py
import pytest
import numpy as np
from bci_processor.feature_extraction import FeatureExtractor
from bci_processor.utils import load_config

@pytest.fixture
def config():
    return load_config("bci_processor/config.yaml")

@pytest.fixture
def extractor(config):
    return FeatureExtractor(config)

def test_extract_wavelet_features(extractor):
    signal = np.random.randn(2250)
    features = extractor.extract_wavelet_features(signal, wavelet='db4', levels=5)
    assert len(features) > 0
    assert all(np.isfinite(features))

def test_extract_mi_features(extractor):
    eeg_data = np.random.randn(2250, 8)
    processed_data = [eeg_data] * len(extractor.mi_config['wavelet_levels'])
    features = extractor.extract_mi_features(eeg_data, processed_data)
    assert len(features) > 0
    assert all(np.isfinite(features))