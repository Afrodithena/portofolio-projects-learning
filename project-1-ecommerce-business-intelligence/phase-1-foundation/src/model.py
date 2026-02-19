import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeliveryTimePredictor:
    """
    Random Forest model for predicting delivery time in days.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, min_samples_leaf: int = 3,
                 random_state: int = 42, n_jobs: int = -1):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.is_trained = False
        self.feature_names = None

        logger.info(f"DeliveryTimePredictor initialized")
        logger.info(f"n_estimators: {n_estimators}, max_depth: {max_depth}")
    
    def _initialize_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs       
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[list[str]] = None) -> Dict[str, float]:
        logger.info(f"Training with X shape: {X.shape}, y shape: {y.shape}")

        if feature_names is None:
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = feature_names
        
        self.model = self._initialize_model()
        self.model.fit(X, y)
        self.is_trained = True
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'r2_train': r2_score(y, y_pred),
            'mae_train': mean_absolute_error(y, y_pred),
            'rmse_train': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"Training completed - R2: {metrics['r2_train']:.4f}, MAE: {metrics['mae_train']:.2f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(X)} samples")
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating on test data: {X_test.shape}")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'r2_test': r2_score(y_test, y_pred),
            'mae_test': mean_absolute_error(y_test, y_pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        logger.info(f"Evaluation results - R2: {metrics['r2_test']:.4f}, MAE: {metrics['mae_test']:.2f}")
        
        return metrics
    
    def save(self, filepath: str = None) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if filepath is None:
            filepath = "D:/portofolio-projects-learning/project-1-ecommerce-business-intelligence/phase-1-foundation/models/delivery_model.pkl"
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str = None) -> None:        
        if filepath is None:
            filepath = "D:/portofolio-projects-learning/project-1-ecommerce-business-intelligence/phase-1-foundation/models/delivery_model.pkl"
        
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names', None)
        self.is_trained = True
        
        if 'hyperparameters' in model_data:
            hp = model_data['hyperparameters']
            self.n_estimators = hp.get('n_estimators', self.n_estimators)
            self.max_depth = hp.get('max_depth', self.max_depth)
            self.min_samples_split = hp.get('min_samples_split', self.min_samples_split)
            self.min_samples_leaf = hp.get('min_samples_leaf', self.min_samples_leaf)
            self.random_state = hp.get('random_state', self.random_state)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained or loaded")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        importance['importance_pct'] = (importance['importance'] / importance['importance'].sum() * 100).round(2)
        
        logger.info("Top 5 most important features:")
        for i in range(min(5, len(importance))):
            logger.info(f"{i+1}. {importance.iloc[i]['feature']}: {importance.iloc[i]['importance_pct']:.2f}%")
        
        return importance


if __name__ == "__main__":
    print("Testing DeliveryTimePredictor")
    
    np.random.seed(42)
    n_samples = 1000
    
    X_test = pd.DataFrame({
        'total_price': np.random.uniform(50, 1000, n_samples),
        'n_items': np.random.randint(1, 5, n_samples),
        'total_freight': np.random.uniform(10, 100, n_samples),
        'purchase_month': np.random.randint(1, 13, n_samples)
    })
    
    y_test = (X_test['total_price'] / 100 + 
              X_test['n_items'] * 0.5 + 
              X_test['total_freight'] / 10 +
              np.random.normal(0, 2, n_samples))

    from sklearn.model_selection import train_test_split
    
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42
    )
    
    model = DeliveryTimePredictor(n_estimators=50, max_depth=5)
    
    train_metrics = model.train(X_train, y_train)
    print(f"Train metrics: {train_metrics}")
    
    eval_metrics = model.evaluate(X_test_split, y_test_split)
    print(f"Evaluation metrics: {eval_metrics}")

    importance = model.get_feature_importance()
    print(importance.head())

    model.save()
    
    model2 = DeliveryTimePredictor()
    model2.load()
    
    print("Model test successful!")