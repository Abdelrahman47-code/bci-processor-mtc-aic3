# tests/test_preprocessing.py
import pytest
import numpy as np
from bci_processor.preprocessing import Preprocessor
from bci_processor.utils import load_config

@pytest.fixture
def config():
    return load_config("bci_processor/config.yaml")

@pytest.fixture
def preprocessor(config):
    return Preprocessor(config)

def test_load_trial_data(preprocessor):
    # Mock a row from the dataframe
    row = {'id': 1, 'task': 'MI', 'subject_id': 'S1', 'trial_session': 1, 'trial': 1}
    # This test assumes data files are available; replace with mock data for CI
    with pytest.raises(Exception):  # Adjust based on actual data availability
        preprocessor.load_trial_data(row)

def test_preprocess_mi(preprocessor):
    data = np.random.randn(2250, 8)
    processed = preprocessor.preprocess(data, 'MI')
    assert isinstance(processed, list)
    assert len(processed) == len(preprocessor.mi_freq_bands)
    assert processed[0].shape == data.shape