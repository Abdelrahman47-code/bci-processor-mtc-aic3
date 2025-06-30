# BCI Processor

A professional Python package for processing and classifying EEG signals for Brain-Computer Interface (BCI) applications, supporting Motor Imagery (MI) and Steady-State Visually Evoked Potential (SSVEP) tasks.

## Features
- Modular design with separate preprocessing, feature extraction, and modeling components.
- Advanced preprocessing: ICA, notch filtering, bandpass filtering, subject-specific normalization.
- Feature extraction: FBCSP for MI, extended CCA for SSVEP, wavelet features, SNR, and phase features.
- Models: LDA for MI, Gradient Boosting with hyperparameter tuning for SSVEP.
- Configurable via YAML file.
- Logging and error handling for robustness.
- Unit tests with pytest.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bci-processor.git
   cd bci-processor