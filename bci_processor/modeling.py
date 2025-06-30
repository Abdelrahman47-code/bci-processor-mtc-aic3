# bci_processor/modeling.py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
from typing import Tuple, Optional

class BCIModel:
    """Manages training and prediction for MI and SSVEP models."""
    
    def __init__(self, config: dict):
        self.mi_config = config['modeling']['mi']
        self.ssvep_config = config['modeling']['ssvep']
        self.scaler_mi = RobustScaler()
        self.scaler_ssvep = RobustScaler()
        self.feature_selector_mi = SelectKBest(f_classif, k=config['feature_extraction']['mi']['k_features'])
        self.feature_selector_ssvep = SelectKBest(f_classif, k=config['feature_extraction']['ssvep']['k_features'])
        self.mi_model = None
        self.ssvep_model = None
    
    def train_mi_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train MI model and return validation accuracy."""
        X_train_selected = self.feature_selector_mi.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector_mi.transform(X_val)
        
        X_train_scaled = self.scaler_mi.fit_transform(X_train_selected)
        X_val_scaled = self.scaler_mi.transform(X_val_selected)
        
        self.mi_model = LinearDiscriminantAnalysis(
            solver=self.mi_config['solver'],
            shrinkage=self.mi_config['shrinkage']
        )
        self.mi_model.fit(X_train_scaled, y_train)
        
        val_pred = self.mi_model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, val_pred)
        logging.info(f"MI Validation Accuracy: {accuracy:.4f}")
        logging.info("\nMI Classification Report:\n" + classification_report(y_val, val_pred))
        return accuracy
    
    def train_ssvep_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train SSVEP model with hyperparameter tuning and return validation accuracy."""
        X_train_selected = self.feature_selector_ssvep.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector_ssvep.transform(X_val)
        
        X_train_scaled = self.scaler_ssvep.fit_transform(X_train_selected)
        X_val_scaled = self.scaler_ssvep.transform(X_val_selected)
        
        self.ssvep_model = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_distributions=self.ssvep_config['param_dist'],
            n_iter=self.ssvep_config['n_iter'],
            cv=self.ssvep_config['cv'],
            scoring='accuracy',
            n_jobs=-1
        )
        self.ssvep_model.fit(X_train_scaled, y_train)
        
        logging.info(f"Best SSVEP parameters: {self.ssvep_model.best_params_}")
        val_pred = self.ssvep_model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, val_pred)
        logging.info(f"SSVEP Validation Accuracy: {accuracy:.4f}")
        logging.info("\nSSVEP Classification Report:\n" + classification_report(y_val, val_pred))
        
        if hasattr(self.ssvep_model.best_estimator_, 'feature_importances_'):
            feature_importance = self.ssvep_model.best_estimator_.feature_importances_
            top_features = np.argsort(feature_importance)[-10:][::-1]
            logging.info("\nTop 10 SSVEP features:")
            for i, feat_idx in enumerate(top_features):
                logging.info(f"{i+1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
        
        return accuracy
    
    def predict(self, task_type: str, features: np.ndarray) -> str:
        """Predict label for a single trial."""
        try:
            if task_type == 'MI':
                features_selected = self.feature_selector_mi.transform(features.reshape(1, -1))
                features_scaled = self.scaler_mi.transform(features_selected)
                return self.mi_model.predict(features_scaled)[0]
            else:
                features_selected = self.feature_selector_ssvep.transform(features.reshape(1, -1))
                features_scaled = self.scaler_ssvep.transform(features_selected)
                return self.ssvep_model.best_estimator_.predict(features_scaled)[0]
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 'Left' if task_type == 'MI' else 'Forward'