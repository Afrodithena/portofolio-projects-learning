import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DeliveryTimePredictor:
    """
    Predictor class for delivery time using trained Random Forest model.
    """
    
    def __init__(self, model_path: str = None):    
        if model_path is None:
            model_path = "D:/portofolio-projects-learning/project-1-ecommerce-business-intelligence/phase-1-foundation/models/delivery_model.pkl"
        
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from file."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not self.model_path.exists():   
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):   
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', None)
            else:
                self.model = model_data      
                self.feature_names = None    
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully")
            
            if self.feature_names:
                logger.info(f"Expected features: {self.feature_names[:5]}...")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, input_data: Dict) -> Dict:
        """
        Make prediction for a single order.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction result
        """
        if not self.is_loaded:
            raise ValueError(f"Model not loaded. Call _load_model() first")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure features are in correct order
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            df = df[self.feature_names]

        # Make prediction
        prediction = self.model.predict(df)[0]
        
        result = {
            'prediction': round(float(prediction), 2),
            'unit': 'days',
            'input_summary': {
                'total_price': input_data.get('total_price', 0),
                'n_items': input_data.get('n_items', 0)
            }
        }

        logger.info(f"Prediction: {result['prediction']} days")
        return result
    
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple orders.
        
        Args:
            input_list: List of dictionaries with feature values
            
        Returns:
            List of prediction results
        """
        results = []
        
        for data in input_list:
            try:
                res = self.predict_single(data)
                results.append(res)
            except Exception as e:
                logger.error(f"Error predicting for {data}: {e}")
                results.append({'prediction': None, 'error': str(e), 'input': data})
                
        return results


if __name__ == "__main__":
    print("Testing DeliveryTimePredictor")
    
    # Initialize predictor
    predictor = DeliveryTimePredictor()
    
    # Test single prediction
    test_input = {
        'total_price': 500.0,
        'n_items': 3,
        'total_freight': 45.0,
        'purchase_month': 11
    }
    
    result = predictor.predict_single(test_input)      
    print(f"\nSingle prediction result:")
    print(f"Predicted delivery: {result['prediction']} {result['unit']}")

    # Test batch prediction
    test_batch = [
        {'total_price': 100.0, 'n_items': 1, 'total_freight': 10.0, 'purchase_month': 1},
        {'total_price': 750.0, 'n_items': 4, 'total_freight': 60.0, 'purchase_month': 12},
        {'total_price': 300.0, 'n_items': 2, 'total_freight': 25.0, 'purchase_month': 6}
    ]
    
    batch_result = predictor.predict_batch(test_batch) 
    print(f"\nBatch prediction results:")
    
    for i, res in enumerate(batch_result):
        if res.get('prediction') is not None:
            print(f"  Input {i+1}: {res['prediction']} days")
        else:
            print(f"  Input {i+1}: ERROR - {res.get('error')}")
    
    print("\nPredictor test successful")