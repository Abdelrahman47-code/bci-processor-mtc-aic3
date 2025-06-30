# tests/test_modeling.py
import pytest
import numpy as np
from bci_processor.modeling import BCIModel
from bci_processor.utils import load_config

@pytest.fixture
def config():
    return load_config("bci_processor/config.yaml")

@pytest.fixture
def model(config):
    return BCIModel(config)

def test_train_mi_model(model):
    X_train = np.random.randn(100, 60)
    y_train = np.random.choice(['Left', 'Right'], 100)
    X_val = np.random.randn(20, 60)
    y_val = np.random.choice(['Left', 'Right'], 20)
    accuracy = model.train_mi_model(X_train, y_train, X_val, y_val)
    assert 0 <= accuracy <= 1