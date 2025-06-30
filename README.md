# BCI Processor

A modular Python package for processing and classifying EEG signals for Brain-Computer Interface (BCI) applications, developed for the MTC-AIC-3: Egypt National Artificial Intelligence Competition. The system supports Motor Imagery (MI) and Steady-State Visually Evoked Potential (SSVEP) tasks, achieving robust performance through advanced preprocessing, feature extraction, and machine learning techniques.

## Overview

The `bci-processor` package provides an end-to-end pipeline for EEG signal classification, designed for the MTC-AIC-3 competition. It processes EEG data from the competition dataset, extracts features using Filter Bank Common Spatial Patterns (FBCSP) for MI and extended Canonical Correlation Analysis (CCA) for SSVEP, and trains models using Linear Discriminant Analysis (LDA) for MI and Gradient Boosting with hyperparameter tuning for SSVEP. The system is fully reproducible, configurable via a YAML file, and includes unit tests and model checkpoints.

### Key Features
- **Modular Design**: Separate modules for preprocessing, feature extraction, and modeling.
- **Preprocessing**: Subject-specific normalization, ICA-based artifact removal, notch and bandpass filtering.
- **Feature Extraction**:
  - MI: Time-domain, frequency-domain, wavelet, FBCSP, and cross-channel features.
  - SSVEP: Frequency-domain, extended CCA, SNR, and phase features.
- **Modeling**: LDA for MI, Gradient Boosting with RandomizedSearchCV for SSVEP.
- **Configuration**: Parameters managed via `config.yaml`.
- **Reproducibility**: Includes model checkpoints, dependencies, and a CLI script.
- **Testing**: Unit tests with `pytest` for validation.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/bci-processor.git
   cd bci-processor
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Package**:
   ```bash
   pip install .
   ```

## Usage

### Running the Pipeline
The pipeline can be executed via the command-line interface (CLI) script:
```bash
python scripts/run_bci_processor.py --config bci_processor/config.yaml --output submission.csv --log-level INFO
```

This command:
- Trains MI and SSVEP models on the training data.
- Generates predictions for the test set in `submission.csv`.
- Logs progress and results to the console and `bci_processor.log`.

### Configuration
Edit `bci_processor/config.yaml` to customize parameters:
- **Data**: Dataset path, sampling rate, EEG channels.
- **Preprocessing**: Frequency bands, notch filter settings, ICA components.
- **Feature Extraction**: Wavelet settings, CSP filters, target frequencies for SSVEP.
- **Modeling**: Model types and hyperparameter search space.


### Testing
Run unit tests to validate the code:
```bash
pytest tests/
```

## Repository Structure
```
bci-processor/
├── bci_processor/          # Core Python package
│   ├── __init__.py
│   ├── config.yaml         # Configuration file
│   ├── preprocessing.py    # Data loading and preprocessing
│   ├── feature_extraction.py # Feature extraction for MI and SSVEP
│   ├── modeling.py         # Model training and prediction
│   ├── processor.py        # Main integration class
│   └── utils.py            # Utility functions (logging, config loading)
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_modeling.py
├── scripts/                # CLI scripts
│   └── run_bci_processor.py
├── .gitignore              # Git ignore file
├── README.md               # This file
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── setup.py               # Package installation script
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions for CI/CD
```


## Performance
Approximate validation accuracies:
- **MI**: 0.6000
- **SSVEP**: 0.4600
- **Overall**: 0.5300

## Challenges and Solutions
- **CSP Indexing Error**: Fixed by using trial-specific covariance matrices, removing label dependency during testing.
- **Noisy EEG Data**: Addressed with ICA-based artifact removal and robust scaling.
- **SSVEP Performance**: Improved with extended CCA (phase-shifted references) and hyperparameter tuning.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
- **Team Name**: Synaptic Squad
- **Team Leader**: Abdelrahman Ahmed Mahmoud Eldaba
- **Email**: abdelrahmaneldaba123@gmail.com

Thank you for exploring `bci-processor`! We hope this package provides a robust foundation for EEG-based BCI research.
