import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import OlistDataLoader
from model import DeliveryTimePredictor

# Setup logging
log_path = "D:/portofolio-projects-learning/project-1-ecommerce-business-intelligence/phase-1-foundation/logs/trainer.log"
import os
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('../logs/trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeliveryPredictorTrainer:
    """
    Trainer class for delivery time prediction model.
    Handles data preparation, model training, evaluation, and saving.
    """
    
    def __init__(self,
                 data_path: str = "../data/raw",
                 model_save_path: str = "../models",
                 test_size: float = 0.2,
                 random_state: int = 42):
        
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.test_size = test_size
        self.random_state = random_state

        # Create model directory if not exists
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = OlistDataLoader(data_path)
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.training_results = {}

        logger.info(f"Delivery predictor trainer initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Model save path: {self.model_save_path}")
        logger.info(f"Test size: {test_size}")

    def prepare_data(self):
        """Load, clean, and prepare data for training."""
        logger.info("Loading Preparing Data...")
        raw_data = self.loader.load_all_raw()
        
        logger.info("\nCleaning dataset...")
        raw_data['orders'] = self.loader.clean_orders(raw_data['orders'])
        raw_data['reviews'] = self.loader.clean_reviews(raw_data['reviews'])
        raw_data['products'] = self.loader.clean_products(raw_data['products'])

        logger.info("\nEngineering Features...")
        df_features = self.loader.engineer_features(raw_data)

        logger.info("\nPreparing for modeling...")
        X, y, feature_names = self.loader.prepare_for_modeling(df_features)

        logger.info(f"\nSplitting data (test_size: {self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

        logger.info(f"Data preparation complete:")
        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train_model(self, 
                    n_estimators: int = 100,
                    max_depth: int = 10,
                    min_samples_split: int = 5,
                    min_samples_leaf: int = 2) -> DeliveryTimePredictor:
        
        logger.info("Training Model")
        
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first")
            
        logger.info(f"\nInitializing model with:")
        logger.info(f"n_estimators: {n_estimators}")
        logger.info(f"max_depth: {max_depth}")
        logger.info(f"min_samples_split: {min_samples_split}")
        logger.info(f"min_samples_leaf: {min_samples_leaf}")

        self.model = DeliveryTimePredictor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )

        # Train model
        train_metrics = self.model.train(
            self.X_train, self.y_train, feature_names=self.feature_names
        )

        # Evaluate on test set
        eval_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Store results
        self.training_results = {
            'train': train_metrics,
            'test': eval_metrics,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            },
            'data_shape': {
                'train': self.X_train.shape,
                'test': self.X_test.shape,
                'features': len(self.feature_names)
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("\nTraining complete:")
        logger.info(f"Train MAE: {train_metrics['mae_train']:.2f} days")
        logger.info(f"Test MAE: {eval_metrics['mae_test']:.2f} days")
        logger.info(f"Test R2: {eval_metrics['r2_test']:.3f}")
    
        return self.model
    
    def save_model(self, filename: str = None) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"delivery_model_{timestamp}.pkl"
        
        save_path = self.model_save_path / filename
        self.model.save(str(save_path))
        logger.info(f"Model saved to: {save_path}")
        
        return str(save_path)
    
    def generate_report(self) -> dict:
        """Generate training report with metrics and feature importance."""
        if self.model is None:
            raise ValueError("No model trained yet.")
            
        importance = self.model.get_feature_importance()
        
        report = {
            'summary': {
                'model_type': 'RandomForestRegressor',
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape,
                'test_size': self.test_size
            },
            'performance': {
                'train': self.training_results.get('train', {}),
                'test': self.training_results.get('test', {})
            },
            'hyperparameters': self.training_results.get('hyperparameters', {}),
            'feature_importance': importance.to_dict('records'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')     
        }

        logger.info("\nTraining report generated")
        return report
    
    def run_full_pipeline(self) -> DeliveryTimePredictor:
        """Run complete training pipeline from data prep to model saving."""
        logger.info("Starting full training pipeline")
        
        self.prepare_data()
        self.train_model()
        self.save_model()
        report = self.generate_report()

        logger.info("Full pipeline completed")
        logger.info(f"Final Test MAE: {report['performance']['test']['mae_test']:.2f} days")
        logger.info(f"Final Test R2: {report['performance']['test']['r2_test']:.3f}")

        return self.model


if __name__ == "__main__":
    print("Testing delivery trainer")
    
    trainer = DeliveryPredictorTrainer(
        data_path="../data/raw", 
        model_save_path="../models", 
        test_size=0.2
    )
    
    model = trainer.run_full_pipeline()
    
    print("\nTop 5 features:")
    importance = model.get_feature_importance()
    print(importance.head())
    
    print("\nTrainer test successful")