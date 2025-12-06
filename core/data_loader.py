"""
Data Loader Module
Loads and preprocesses product and review data from CSV files

This module provides functions to load product catalogs and user reviews,
with built-in validation to ensure data integrity.
"""

from typing import Optional
import pandas as pd
from pathlib import Path


def load_products(path: str) -> pd.DataFrame:
    """
    Load product catalog data from CSV file.
    
    Expected CSV columns:
    - product_id: int
    - name: str
    - category_id: int
    - price: float
    - avg_rating: float
    
    Args:
        path: Path to the products CSV file
    
    Returns:
        DataFrame with product data and correct dtypes
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    
    Example:
        >>> products = load_products("data/products.csv")
        >>> print(products.head())
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Products file not found: {path}")
    
    try:
        # Load CSV with explicit dtypes
        df = pd.read_csv(
            path,
            dtype={
                'product_id': int,
                'name': str,
                'category_id': int,
                'price': float,
                'avg_rating': float
            }
        )
        
        # Validate required columns
        required_columns = ['product_id', 'name', 'category_id', 'price', 'avg_rating']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns in products CSV: {missing_columns}"
            )
        
        print(f"[DataLoader] Loaded {len(df)} products from {path}")
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading products from {path}: {e}")


def load_reviews(path: str) -> pd.DataFrame:
    """
    Load product reviews data from CSV file.
    
    Expected CSV columns:
    - review_id: int
    - product_id: int
    - rating: int (1-5)
    - review_text: str
    
    Args:
        path: Path to the reviews CSV file
    
    Returns:
        DataFrame with review data and correct dtypes
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing or data is invalid
    
    Example:
        >>> reviews = load_reviews("data/reviews.csv")
        >>> print(reviews.head())
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")
    
    try:
        # Load CSV with explicit dtypes
        df = pd.read_csv(
            path,
            dtype={
                'review_id': int,
                'product_id': int,
                'rating': int,
                'review_text': str
            }
        )
        
        # Validate required columns
        required_columns = ['review_id', 'product_id', 'rating', 'review_text']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns in reviews CSV: {missing_columns}"
            )
        
        # Validate rating range (1-5)
        invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]
        if not invalid_ratings.empty:
            raise ValueError(
                f"Invalid ratings found (must be 1-5): "
                f"{invalid_ratings['review_id'].tolist()}"
            )
        
        print(f"[DataLoader] Loaded {len(df)} reviews from {path}")
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading reviews from {path}: {e}")


def validate_data(products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """
    Validate product and review data integrity.
    
    Performs the following checks:
    1. No missing values in critical columns
    2. All review product_ids exist in products
    3. Data type consistency
    
    Args:
        products: DataFrame containing product data
        reviews: DataFrame containing review data
    
    Raises:
        ValueError: If validation fails with detailed error message
    
    Example:
        >>> products = load_products("data/products.csv")
        >>> reviews = load_reviews("data/reviews.csv")
        >>> validate_data(products, reviews)
        >>> # Raises ValueError if data is invalid
    """
    errors = []
    
    # Check for missing values in products
    products_missing = products.isnull().sum()
    if products_missing.any():
        missing_info = products_missing[products_missing > 0].to_dict()
        errors.append(f"Missing values in products: {missing_info}")
    
    # Check for missing values in reviews
    reviews_missing = reviews.isnull().sum()
    if reviews_missing.any():
        missing_info = reviews_missing[reviews_missing > 0].to_dict()
        errors.append(f"Missing values in reviews: {missing_info}")
    
    # Check that all review product_ids exist in products
    product_ids = set(products['product_id'])
    review_product_ids = set(reviews['product_id'])
    
    orphaned_reviews = review_product_ids - product_ids
    if orphaned_reviews:
        errors.append(
            f"Found {len(orphaned_reviews)} reviews for non-existent products: "
            f"{sorted(list(orphaned_reviews))}"
        )
    
    # Check for duplicate product IDs
    duplicate_products = products[products.duplicated(subset=['product_id'], keep=False)]
    if not duplicate_products.empty:
        dup_ids = duplicate_products['product_id'].unique().tolist()
        errors.append(f"Duplicate product IDs found: {dup_ids}")
    
    # Check for duplicate review IDs
    duplicate_reviews = reviews[reviews.duplicated(subset=['review_id'], keep=False)]
    if not duplicate_reviews.empty:
        dup_ids = duplicate_reviews['review_id'].unique().tolist()
        errors.append(f"Duplicate review IDs found: {dup_ids}")
    
    # Check price validity
    invalid_prices = products[products['price'] <= 0]
    if not invalid_prices.empty:
        errors.append(
            f"Invalid prices (must be > 0): "
            f"product_ids {invalid_prices['product_id'].tolist()}"
        )
    
    # Check rating validity
    invalid_avg_ratings = products[
        (products['avg_rating'] < 0) | (products['avg_rating'] > 5)
    ]
    if not invalid_avg_ratings.empty:
        errors.append(
            f"Invalid average ratings (must be 0-5): "
            f"product_ids {invalid_avg_ratings['product_id'].tolist()}"
        )
    
    # If any errors found, raise ValueError with all issues
    if errors:
        error_message = "\n".join([f"  - {error}" for error in errors])
        raise ValueError(f"Data validation failed:\n{error_message}")
    
    print("[DataLoader] ✓ Data validation passed")
    print(f"[DataLoader]   - {len(products)} products validated")
    print(f"[DataLoader]   - {len(reviews)} reviews validated")
    print(f"[DataLoader]   - All review product_ids exist in products")


def get_data_summary(products: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the loaded data.
    
    Args:
        products: DataFrame containing product data
        reviews: DataFrame containing review data
    
    Returns:
        Dictionary with summary statistics
    
    Example:
        >>> summary = get_data_summary(products, reviews)
        >>> print(f"Average price: ${summary['avg_price']:.2f}")
    """
    summary = {
        'num_products': len(products),
        'num_reviews': len(reviews),
        'num_categories': products['category_id'].nunique(),
        'avg_price': products['price'].mean(),
        'min_price': products['price'].min(),
        'max_price': products['price'].max(),
        'avg_rating': products['avg_rating'].mean(),
        'reviews_per_product': len(reviews) / len(products) if len(products) > 0 else 0,
        'price_range': products['price'].max() - products['price'].min()
    }
    
    return summary


if __name__ == "__main__":
    """
    Demonstration of data loading and validation.
    
    Run this script directly to test the data loader with sample CSVs:
        python core/data_loader.py
    """
    import os
    
    print("=" * 70)
    print("PaReCo-Py Data Loader Demo")
    print("=" * 70)
    
    # Determine paths relative to this file
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    
    products_path = data_dir / "products.csv"
    reviews_path = data_dir / "reviews.csv"
    
    print(f"\nLoading data from:")
    print(f"  Products: {products_path}")
    print(f"  Reviews:  {reviews_path}")
    print()
    
    try:
        # Load data
        products_df = load_products(str(products_path))
        reviews_df = load_reviews(str(reviews_path))
        
        print()
        
        # Validate data
        validate_data(products_df, reviews_df)
        
        print()
        print("-" * 70)
        print("Data Summary")
        print("-" * 70)
        
        # Get and display summary
        summary = get_data_summary(products_df, reviews_df)
        
        print(f"Products:              {summary['num_products']}")
        print(f"Reviews:               {summary['num_reviews']}")
        print(f"Categories:            {summary['num_categories']}")
        print(f"Average Price:         ${summary['avg_price']:.2f}")
        print(f"Price Range:           ${summary['min_price']:.2f} - ${summary['max_price']:.2f}")
        print(f"Average Rating:        {summary['avg_rating']:.2f}/5.0")
        print(f"Reviews per Product:   {summary['reviews_per_product']:.1f}")
        
        print()
        print("-" * 70)
        print("Sample Products (first 5)")
        print("-" * 70)
        print(products_df.head().to_string(index=False))
        
        print()
        print("-" * 70)
        print("Sample Reviews (first 5)")
        print("-" * 70)
        print(reviews_df.head().to_string(index=False))
        
        print()
        print("=" * 70)
        print("✓ Data loading and validation successful!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import sys
        sys.exit(1)
