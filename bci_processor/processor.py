# bci_processor/processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from .preprocessing import Preprocessor
from .feature_extraction import FeatureExtractor
from .modeling import BCIModel
from .utils import load_config

class EnhancedBCIProcessor:
    """Main class for BCI signal processing and classification."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize processor with configuration."""
        self.config = load_config(config_path)
        self.preprocessor = Preprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = BCIModel(self.config)
        
        # Load data
        self.train_df = pd.read_csv(self.config['data']['base_path'] + '/train.csv')
        self.validation_df = pd.read_csv(self.config['data']['base_path'] + '/validation.csv')
        self.test_df = pd.read_csv(self.config['data']['base_path'] + '/test.csv')
    
    def prepare_dataset(self, df: pd.DataFrame, task_type: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features and labels for a dataset."""
        task_data = df[df['task'] == task_type].copy()
        features = []
        labels = []
        
        logging.info(f"Processing {len(task_data)} {task_type} trials...")
        for idx, row in task_data.iterrows():
            try:
                eeg_data = self.preprocessor.load_trial_data(row)
                min_samples = self.config['data']['ssvep_samples_per_trial'] if task_type == 'SSVEP' else self.config['data']['mi_samples_per_trial']
                if eeg_data.shape[0] < min_samples:
                    logging.warning(f"Skipping trial {row['id']}: insufficient data ({eeg_data.shape[0]} < {min_samples})")
                    continue
                
                processed_data = self.preprocessor.preprocess(eeg_data, task_type)
                if task_type == 'MI':
                    feature_vector = self.feature_extractor.extract_mi_features(eeg_data, processed_data, row.get('label'))
                else:
                    feature_vector = self.feature_extractor.extract_ssvep_features(eeg_data, processed_data)
                
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                features.append(feature_vector)
                
                if 'label' in row:
                    labels.append(row['label'])
                
                if len(features) % 100 == 0:
                    logging.info(f"Processed {len(features)} trials...")
            except Exception as e:
                logging.error(f"Error processing trial {row['id']}: {e}")
                continue
        
        logging.info(f"Successfully processed {len(features)} {task_type} trials")
        return np.array(features), np.array(labels) if labels else None
    
    def train(self) -> tuple[float, float]:
        """Train MI and SSVEP models and return validation accuracies."""
        X_mi_train, y_mi_train = self.prepare_dataset(self.train_df, 'MI')
        X_mi_val, y_mi_val = self.prepare_dataset(self.validation_df, 'MI')
        mi_accuracy = self.model.train_mi_model(X_mi_train, y_mi_train, X_mi_val, y_mi_val)
        
        X_ssvep_train, y_ssvep_train = self.prepare_dataset(self.train_df, 'SSVEP')
        X_ssvep_val, y_ssvep_val = self.prepare_dataset(self.validation_df, 'SSVEP')
        ssvep_accuracy = self.model.train_ssvep_model(X_ssvep_train, y_ssvep_train, X_ssvep_val, y_ssvep_val)
        
        overall_accuracy = (mi_accuracy + ssvep_accuracy) / 2
        logging.info(f"Overall Validation Accuracy: {overall_accuracy:.4f}")
        return mi_accuracy, ssvep_accuracy
    
    def generate_predictions(self, output_path: str = "enhanced_submission.csv") -> pd.DataFrame:
        """Generate predictions for test set."""
        predictions = []
        logging.info("Generating test predictions...")
        
        for idx, row in self.test_df.iterrows():
            try:
                eeg_data = self.preprocessor.load_trial_data(row)
                processed_data = self.preprocessor.preprocess(eeg_data, row['task'])
                features = self.feature_extractor.extract_mi_features(eeg_data, processed_data) if row['task'] == 'MI' else \
                          self.feature_extractor.extract_ssvep_features(eeg_data, processed_data)
                prediction = self.model.predict(row['task'], features)
                predictions.append({'id': row['id'], 'label': prediction})
            except Exception as e:
                logging.error(f"Error predicting trial {row['id']}: {e}")
                fallback = 'Left' if row['task'] == 'MI' else 'Forward'
                predictions.append({'id': row['id'], 'label': fallback})
        
        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv(output_path, index=False)
        logging.info(f"Submission file saved with {len(predictions)} predictions at {output_path}")
        return submission_df