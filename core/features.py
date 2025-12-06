"""
Feature Engineering Module
Extracts and transforms features for recommendations

This module provides the FeatureBuilder class for creating normalized
numeric feature matrices from product data.
"""

from typing import Any
import numpy as np
import pandas as pd
from core.device_manager import DeviceManager


class FeatureBuilder:
    """
    Builds numeric feature vectors for products to be used by recommendation engine.
    
    Extracts product attributes (price, rating, category) and creates normalized
    feature matrices that can be used for similarity calculations and recommendations.
    Uses DeviceManager to create device-agnostic tensors (NumPy or PyTorch).
    
    Attributes:
        device_manager (DeviceManager): Manager for device-specific tensor operations
    
    Example:
        >>> from core.device_manager import DeviceManager
        >>> from core.data_loader import load_products
        >>> 
        >>> dm = DeviceManager()
        >>> fb = FeatureBuilder(dm)
        >>> products = load_products("data/products.csv")
        >>> features = fb.build_product_matrix(products)
        >>> print(features.shape)  # (num_products, 3)
    """
    
    def __init__(self, device_manager: DeviceManager) -> None:
        """
        Initialize FeatureBuilder with a DeviceManager.
        
        Args:
            device_manager: DeviceManager instance for tensor operations
        
        Example:
            >>> dm = DeviceManager()
            >>> fb = FeatureBuilder(dm)
        """
        self.device_manager = device_manager
        print(f"[FeatureBuilder] Initialized with device: {device_manager.device}")
    
    def build_product_matrix(self, products_df: pd.DataFrame) -> Any:
        """
        Build normalized feature matrix for products.
        
        Creates a feature matrix with shape (num_products, 3) containing:
        - Column 0: Normalized price (standardized)
        - Column 1: Normalized average rating (standardized)
        - Column 2: Category ID (as float)
        
        Standardization formula: (x - mean) / (std + 1e-9)
        
        Args:
            products_df: DataFrame with columns: product_id, category_id, price, avg_rating
        
        Returns:
            Feature matrix as np.ndarray or torch.Tensor (depending on device)
            Shape: (num_products, 3)
        
        Raises:
            ValueError: If required columns are missing
        
        Example:
            >>> products = load_products("data/products.csv")
            >>> features = fb.build_product_matrix(products)
            >>> print(f"Feature matrix shape: {features.shape}")
            Feature matrix shape: (15, 3)
        """
        # Validate required columns
        required_cols = ['product_id', 'category_id', 'price', 'avg_rating']
        missing_cols = set(required_cols) - set(products_df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns in products DataFrame: {missing_cols}"
            )
        
        if len(products_df) == 0:
            raise ValueError("Products DataFrame is empty")
        
        # Extract features as NumPy arrays
        price = products_df['price'].to_numpy(dtype=np.float64)
        avg_rating = products_df['avg_rating'].to_numpy(dtype=np.float64)
        category_id = products_df['category_id'].to_numpy(dtype=np.float64)
        
        # Standardize price: (x - mean) / (std + epsilon)
        price_mean = np.mean(price)
        price_std = np.std(price)
        normalized_price = (price - price_mean) / (price_std + 1e-9)
        
        # Standardize average rating: (x - mean) / (std + epsilon)
        rating_mean = np.mean(avg_rating)
        rating_std = np.std(avg_rating)
        normalized_rating = (avg_rating - rating_mean) / (rating_std + 1e-9)
        
        # Stack features into matrix: [normalized_price, normalized_rating, category_id]
        # Shape: (num_products, 3)
        feature_matrix = np.column_stack([
            normalized_price,
            normalized_rating,
            category_id
        ])
        
        print(f"[FeatureBuilder] Built feature matrix with shape {feature_matrix.shape}")
        print(f"[FeatureBuilder]   - Price range: [{price.min():.2f}, {price.max():.2f}]")
        print(f"[FeatureBuilder]   - Rating range: [{avg_rating.min():.2f}, {avg_rating.max():.2f}]")
        print(f"[FeatureBuilder]   - Categories: {int(category_id.min())} to {int(category_id.max())}")
        
        # Convert to device-specific tensor (NumPy array or torch.Tensor)
        features_tensor = self.device_manager.tensor(feature_matrix, dtype="float32")
        
        return features_tensor
    
    def get_product_ids(self, products_df: pd.DataFrame) -> np.ndarray:
        """
        Extract product IDs as a NumPy array.
        
        Useful for mapping feature matrix rows back to product IDs.
        
        Args:
            products_df: DataFrame with 'product_id' column
        
        Returns:
            NumPy array of product IDs (int64)
        
        Raises:
            ValueError: If 'product_id' column is missing
        
        Example:
            >>> products = load_products("data/products.csv")
            >>> product_ids = fb.get_product_ids(products)
            >>> print(product_ids)
            [1 2 3 4 5 ... 15]
        """
        if 'product_id' not in products_df.columns:
            raise ValueError("'product_id' column not found in DataFrame")
        
        if len(products_df) == 0:
            raise ValueError("Products DataFrame is empty")
        
        product_ids = products_df['product_id'].to_numpy(dtype=np.int64)
        
        print(f"[FeatureBuilder] Extracted {len(product_ids)} product IDs")
        
        return product_ids
    
    def get_feature_statistics(self, products_df: pd.DataFrame) -> dict:
        """
        Calculate feature statistics before normalization.
        
        Useful for understanding the data distribution and normalization parameters.
        
        Args:
            products_df: DataFrame with product data
        
        Returns:
            Dictionary with statistics for each feature
        
        Example:
            >>> stats = fb.get_feature_statistics(products)
            >>> print(f"Price mean: ${stats['price']['mean']:.2f}")
        """
        required_cols = ['price', 'avg_rating', 'category_id']
        missing_cols = set(required_cols) - set(products_df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        stats = {
            'price': {
                'mean': products_df['price'].mean(),
                'std': products_df['price'].std(),
                'min': products_df['price'].min(),
                'max': products_df['price'].max()
            },
            'avg_rating': {
                'mean': products_df['avg_rating'].mean(),
                'std': products_df['avg_rating'].std(),
                'min': products_df['avg_rating'].min(),
                'max': products_df['avg_rating'].max()
            },
            'category_id': {
                'unique_count': products_df['category_id'].nunique(),
                'values': sorted(products_df['category_id'].unique().tolist())
            }
        }
        
        return stats
    
    def __repr__(self) -> str:
        """
        String representation of FeatureBuilder.
        
        Returns:
            Formatted string showing configuration
        """
        return (
            f"FeatureBuilder(device='{self.device_manager.device}', "
            f"use_gpu={self.device_manager.use_gpu})"
        )


if __name__ == "__main__":
    """
    Demonstration of FeatureBuilder functionality.
    
    Run this script to test feature extraction:
        python core/features.py
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from core.device_manager import DeviceManager
    from core.data_loader import load_products
    
    print("=" * 70)
    print("PaReCo-Py FeatureBuilder Demo")
    print("=" * 70)
    
    try:
        # Initialize DeviceManager
        print("\n[1] Initializing DeviceManager...")
        dm = DeviceManager()
        print()
        
        # Initialize FeatureBuilder
        print("[2] Initializing FeatureBuilder...")
        fb = FeatureBuilder(dm)
        print()
        
        # Load products
        print("[3] Loading product data...")
        data_dir = current_dir.parent / "data"
        products_path = data_dir / "products.csv"
        products_df = load_products(str(products_path))
        print()
        
        # Get feature statistics
        print("[4] Calculating feature statistics...")
        stats = fb.get_feature_statistics(products_df)
        print(f"  Price:   mean=${stats['price']['mean']:.2f}, "
              f"std=${stats['price']['std']:.2f}")
        print(f"  Rating:  mean={stats['avg_rating']['mean']:.2f}, "
              f"std={stats['avg_rating']['std']:.2f}")
        print(f"  Categories: {stats['category_id']['unique_count']} unique "
              f"({stats['category_id']['values']})")
        print()
        
        # Build feature matrix
        print("[5] Building feature matrix...")
        features = fb.build_product_matrix(products_df)
        print(f"  Feature type: {type(features)}")
        print(f"  Feature shape: {features.shape}")
        print()
        
        # Get product IDs
        print("[6] Extracting product IDs...")
        product_ids = fb.get_product_ids(products_df)
        print(f"  Product IDs: {product_ids[:5]}... ({len(product_ids)} total)")
        print()
        
        # Display sample features
        print("-" * 70)
        print("Sample Feature Matrix (first 5 products)")
        print("-" * 70)
        print("Format: [normalized_price, normalized_rating, category_id]")
        print()
        
        # Convert to numpy for display if needed
        if hasattr(features, 'cpu'):
            # PyTorch tensor
            features_np = features.cpu().numpy()
        else:
            # NumPy array
            features_np = features
        
        for i in range(min(5, len(product_ids))):
            pid = product_ids[i]
            feat = features_np[i]
            orig_price = products_df.iloc[i]['price']
            orig_rating = products_df.iloc[i]['avg_rating']
            orig_cat = products_df.iloc[i]['category_id']
            
            print(f"Product {pid:2d}: [{feat[0]:7.3f}, {feat[1]:7.3f}, {feat[2]:3.0f}]  "
                  f"(original: ${orig_price:6.2f}, {orig_rating:.1f}★, cat={orig_cat})")
        
        print()
        print("=" * 70)
        print("✓ FeatureBuilder demonstration successful!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
