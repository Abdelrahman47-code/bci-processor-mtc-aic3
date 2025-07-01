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