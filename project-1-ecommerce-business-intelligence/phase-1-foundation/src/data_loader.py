import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OlistDataLoader:
    """Class for loading Olist e-commerce data"""
    
    def __init__(self, data_path: str = "../data/raw"):
        self.data_path = Path(data_path)
        logger.info(f"DataLoader initialized with path: {self.data_path.absolute()}")

        if not self.data_path.exists():
            logger.warning(f"Path {self.data_path} does not exist!")
            logger.warning("Make sure running from the correct directory")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.data_path / filename
        logger.info(f"Loading {filename}...")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Shape: {df.shape}")
        return df

    def load_all_raw(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading all raw datasets...")
        
        datasets = {
            'customers': self._load_csv("olist_customers_dataset.csv"),
            'geolocation': self._load_csv("olist_geolocation_dataset.csv"),
            'items': self._load_csv("olist_order_items_dataset.csv"),
            'payments': self._load_csv("olist_order_payments_dataset.csv"),
            'reviews': self._load_csv("olist_order_reviews_dataset.csv"),
            'orders': self._load_csv("olist_orders_dataset.csv"),
            'products': self._load_csv("olist_products_dataset.csv"),
            'sellers': self._load_csv("olist_sellers_dataset.csv"),
            'category_translation': self._load_csv("product_category_name_translation.csv"),
        }

        logger.info("All datasets loaded successfully")
        for name, df in datasets.items():
            logger.info(f"{name:20} : {df.shape}")
        return datasets

    def clean_orders(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning orders dataset...")
        df = orders_df.copy()
        initial_len = len(df)
        df = df.dropna(subset=['order_delivered_customer_date'])
        dropped = initial_len - len(df)

        logger.info(f"Dropped {dropped} rows with missing delivery dates")
        logger.info(f"Final shape: {df.shape}")
        return df

    def clean_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning reviews dataset...")
        df = reviews_df.copy()
        
        if 'review_comment_title' in df.columns:
            df = df.drop('review_comment_title', axis=1)
            logger.info("Dropped 'review_comment_title' (88% missing)")

        if 'review_comment_message' in df.columns:
            df['review_comment_message'] = df['review_comment_message'].fillna('')
            logger.info("Filled missing messages with empty string")
            
        logger.info(f"Final shape: {df.shape}")
        return df

    def clean_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning products dataset...")
        df = products_df.copy()
        
        df['product_category_name'] = df['product_category_name'].fillna('unknown')
        logger.info("Filled missing categories with 'unknown'")
        
        if all(col in df.columns for col in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
            df['product_volume_cm3'] = (
                df['product_length_cm'].fillna(0) * 
                df['product_height_cm'].fillna(0) * 
                df['product_width_cm'].fillna(0)
            ).round(2)
            logger.info("Added 'product_volume_cm3' feature")
        
        logger.info(f"Final shape: {df.shape}")
        return df

    def engineer_features(self, datasets: Dict) -> pd.DataFrame:
        logger.info("Starting feature engineering...")
        
        customers = datasets['customers']
        orders = datasets['orders']
        items = datasets['items']
        products = datasets['products']
        sellers = datasets['sellers']

        logger.info("Creating order-level features...")
        
        order_features = items.groupby('order_id').agg({
            'price': ['sum', 'mean', 'count'],
            'freight_value': ['sum', 'mean']
        }).round(2)
        
        order_features.columns = ['total_price', 'avg_price', 'n_items', 
                                  'total_freight', 'avg_freight']
        order_features = order_features.reset_index()
        
        logger.info(f"Created {len(order_features)} order-level records")
        
        df = orders.merge(order_features, on='order_id', how='left')
        logger.info(f"Shape after order features: {df.shape}")
        
        logger.info("Creating product-level features...")
        
        items_with_products = items.merge(
            products[['product_id', 'product_category_name', 
                      'product_weight_g', 'product_volume_cm3']],
            on='product_id',
            how='left'
        )
        
        items_with_products['product_weight_g'] = items_with_products['product_weight_g'].fillna(0)
        
        product_agg = items_with_products.groupby('order_id').agg({
            'product_weight_g': ['sum', 'mean'],
            'product_volume_cm3': ['sum', 'mean'],
            'product_category_name': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
        }).round(2)
        
        product_agg.columns = ['total_weight', 'avg_weight', 
                               'total_volume', 'avg_volume',
                               'main_category']
        product_agg = product_agg.reset_index()
        
        df = df.merge(product_agg, on='order_id', how='left')
        logger.info(f"Shape after product features: {df.shape}")
        
        logger.info("Creating geographic features...")
        
        df = df.merge(
            customers[['customer_id', 'customer_state', 'customer_city']],
            on='customer_id',
            how='left'
        )
        
        order_seller = items.groupby('order_id')['seller_id'].first().reset_index()
        order_seller = order_seller.merge(
            sellers[['seller_id', 'seller_state', 'seller_city']],
            on='seller_id',
            how='left'
        )
        
        df = df.merge(
            order_seller[['order_id', 'seller_state', 'seller_city']],
            on='order_id',
            how='left'
        )
        
        df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
        df['same_city'] = (df['customer_city'] == df['seller_city']).astype(int)
        
        logger.info(f"Same state rate: {df['same_state'].mean()*100:.1f}%")
        logger.info(f"Shape after geographic features: {df.shape}")
        
        logger.info("Creating temporal features...")
        
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month
        df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
        df['is_weekend'] = (df['purchase_day_of_week'] >= 5).astype(int)
        
        logger.info(f"Weekend purchases: {df['is_weekend'].mean()*100:.1f}%")
        logger.info(f"Shape after temporal features: {df.shape}")

        logger.info("Creating ratio features...")
        
        epsilon = 1e-6
        
        df['freight_to_price_ratio'] = df['total_freight'] / (df['total_price'] + epsilon)
        df['price_per_item'] = df['total_price'] / (df['n_items'] + epsilon)
        df['freight_per_item'] = df['total_freight'] / (df['n_items'] + epsilon)
        
        logger.info(f"Avg freight/price ratio: {df['freight_to_price_ratio'].mean():.3f}")
        logger.info(f"Shape after ratio features: {df.shape}")

        logger.info("Creating target variable...")
        
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
        
        initial_len = len(df)
        df = df[(df['delivery_time_days'] > 0) & (df['delivery_time_days'] < 60)]
        
        logger.info(f"Filtered out {initial_len - len(df)} rows with invalid delivery times")
        logger.info(f"Mean delivery time: {df['delivery_time_days'].mean():.1f} days")
        
        logger.info("Feature engineering complete")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Total features: {len(df.columns)}")
        
        return df

    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        logger.info("Preparing data for modeling...")
        
        exclude_cols = ['order_id', 'customer_id', 'delivery_time_days']
        feature_cols = []
        
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype in ['int64', 'float64']:
                    feature_cols.append(col)
                else:
                    logger.info(f"Skipping non-numeric column: {col} ({df[col].dtype})")
                    
        logger.info(f"Selected {len(feature_cols)} numerical features")

        df_clean = df[feature_cols + ['delivery_time_days']].dropna()
        dropped = len(df) - len(df_clean)
        
        logger.info(f"Dropped {dropped} rows with missing values")
        logger.info(f"Final shape: {df_clean.shape}")
        logger.info(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
        
        X = df_clean[feature_cols]
        y = df_clean['delivery_time_days']
        
        return X, y, feature_cols

# testing
if __name__ == "__main__":
    print("Testing OlistDataLoader")
    
    loader = OlistDataLoader()
    raw_data = loader.load_all_raw()
    raw_data['orders'] = loader.clean_orders(raw_data['orders'])
    raw_data['reviews'] = loader.clean_reviews(raw_data['reviews'])
    raw_data['products'] = loader.clean_products(raw_data['products'])

    df_features = loader.engineer_features(raw_data)
    X, y, features = loader.prepare_for_modeling(df_features)
    
    print("DataLoader test passed")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sample features: {features[:5]}")

    import os
    os.makedirs('data/processed', exist_ok=True)
    df_features.to_csv('data/processed/features_delivery_time.csv', index=False)