"""
Comparator Module
Parallel product comparison engine

This module provides the ParallelComparator class that uses task-based
parallelism via ThreadPoolExecutor to analyze and compare multiple products
in parallel.
"""

from typing import List, Dict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def _analyze_single_product(
    product_id: int,
    products_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> dict:
    """
    Analyze a single product by computing metrics from product data and reviews.
    
    This is a top-level helper function used by ThreadPoolExecutor for parallel
    product analysis.
    
    Args:
        product_id: ID of the product to analyze
        products_df: DataFrame with product information
        reviews_df: DataFrame with all reviews
    
    Returns:
        Dictionary with product analysis:
        - product_id: int
        - name: str
        - price: float
        - avg_rating: float (from reviews)
        - review_count: int
        - positive_ratio: float (fraction of reviews with rating >= 4)
    
    Raises:
        ValueError: If product_id not found in products_df
    
    Example:
        >>> result = _analyze_single_product(1, products_df, reviews_df)
        >>> print(result['positive_ratio'])
        0.75
    """
    # Locate product row
    product_rows = products_df[products_df['product_id'] == product_id]
    
    if product_rows.empty:
        raise ValueError(f"Product ID {product_id} not found in products DataFrame")
    
    product = product_rows.iloc[0]
    
    # Filter reviews for this product
    product_reviews = reviews_df[reviews_df['product_id'] == product_id]
    
    # Compute metrics
    if len(product_reviews) > 0:
        avg_rating = float(product_reviews['rating'].mean())
        review_count = len(product_reviews)
        
        # Positive ratio: fraction with rating >= 4
        positive_reviews = product_reviews[product_reviews['rating'] >= 4]
        positive_ratio = len(positive_reviews) / len(product_reviews)
    else:
        # No reviews - use product's avg_rating if available
        avg_rating = float(product.get('avg_rating', 0.0))
        review_count = 0
        positive_ratio = 0.0
    
    # Build result dictionary
    result = {
        'product_id': int(product_id),
        'name': str(product['name']),
        'price': float(product['price']),
        'avg_rating': avg_rating,
        'review_count': review_count,
        'positive_ratio': positive_ratio
    }
    
    return result


class ParallelComparator:
    """
    Parallel product comparison engine using task-based parallelism.
    
    Uses ThreadPoolExecutor to analyze multiple products concurrently,
    computing metrics like average rating, review count, and positive ratio.
    
    Attributes:
        products_df (pd.DataFrame): Product catalog data
        reviews_df (pd.DataFrame): Product reviews data
    
    Example:
        >>> from core.data_loader import load_products, load_reviews
        >>> products = load_products("data/products.csv")
        >>> reviews = load_reviews("data/reviews.csv")
        >>> comparator = ParallelComparator(products, reviews)
        >>> results = comparator.compare_products([1, 2, 3], n_workers=4)
    """
    
    def __init__(
        self,
        products_df: pd.DataFrame,
        reviews_df: pd.DataFrame
    ) -> None:
        """
        Initialize ParallelComparator with product and review data.
        
        Args:
            products_df: DataFrame with product information
            reviews_df: DataFrame with product reviews
        
        Example:
            >>> comparator = ParallelComparator(products_df, reviews_df)
        """
        self.products_df = products_df
        self.reviews_df = reviews_df
        
        print(f"[ParallelComparator] Initialized with {len(products_df)} products, "
              f"{len(reviews_df)} reviews")
    
    def compare_products(
        self,
        product_ids: List[int],
        n_workers: int = 4
    ) -> List[Dict]:
        """
        Compare multiple products in parallel using task-based parallelism.
        
        Analyzes each product concurrently using ThreadPoolExecutor, computing
        metrics like average rating, review count, and positive review ratio.
        Results are sorted by average rating (descending).
        
        Args:
            product_ids: List of product IDs to compare
            n_workers: Number of parallel worker threads (default: 4)
        
        Returns:
            List of dictionaries with product analysis, sorted by avg_rating
            (highest first). Each dict contains:
            - product_id: int
            - name: str
            - price: float
            - avg_rating: float
            - review_count: int
            - positive_ratio: float
        
        Raises:
            ValueError: If any product_id is not found
        
        Example:
            >>> results = comparator.compare_products([1, 2, 3], n_workers=4)
            >>> for product in results:
            ...     print(f"{product['name']}: {product['avg_rating']:.2f}")
        """
        if not product_ids:
            print("[ParallelComparator] Warning: No product IDs provided")
            return []
        
        print(f"[ParallelComparator] Comparing {len(product_ids)} products "
              f"using {n_workers} workers...")
        
        # Prepare tasks - each product_id gets analyzed independently
        tasks = [
            (pid, self.products_df, self.reviews_df)
            for pid in product_ids
        ]
        
        # Execute parallel analysis using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Map _analyze_single_product over all tasks
            futures = [
                executor.submit(_analyze_single_product, *task)
                for task in tasks
            ]
            
            # Collect results as they complete
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ParallelComparator] Error analyzing product: {e}")
        
        # Sort by average rating (descending)
        results.sort(key=lambda x: x['avg_rating'], reverse=True)
        
        print(f"[ParallelComparator] Successfully compared {len(results)} products")
        
        return results
    
    def compare_by_category(
        self,
        category_id: int,
        n_workers: int = 4
    ) -> List[Dict]:
        """
        Compare all products in a specific category.
        
        Convenience method to compare products within the same category.
        
        Args:
            category_id: Category ID to filter products
            n_workers: Number of parallel workers
        
        Returns:
            List of product analysis dictionaries, sorted by avg_rating
        
        Example:
            >>> results = comparator.compare_by_category(1, n_workers=4)
        """
        # Filter products by category
        category_products = self.products_df[
            self.products_df['category_id'] == category_id
        ]
        
        if category_products.empty:
            print(f"[ParallelComparator] No products found in category {category_id}")
            return []
        
        product_ids = category_products['product_id'].tolist()
        
        print(f"[ParallelComparator] Found {len(product_ids)} products "
              f"in category {category_id}")
        
        return self.compare_products(product_ids, n_workers)
    
    def __repr__(self) -> str:
        """
        String representation of ParallelComparator.
        
        Returns:
            Formatted string showing configuration
        """
        return (
            f"ParallelComparator(n_products={len(self.products_df)}, "
            f"n_reviews={len(self.reviews_df)})"
        )


if __name__ == "__main__":
    """
    Demonstration of ParallelComparator functionality.
    
    Run this script to test product comparison:
        python core/comparator.py
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from core.data_loader import load_products, load_reviews
    
    print("=" * 70)
    print("PaReCo-Py ParallelComparator Demo")
    print("=" * 70)
    
    try:
        # Load data
        print("\n[1] Loading product and review data...")
        data_dir = current_dir.parent / "data"
        products_path = data_dir / "products.csv"
        reviews_path = data_dir / "reviews.csv"
        
        products_df = load_products(str(products_path))
        reviews_df = load_reviews(str(reviews_path))
        print()
        
        # Initialize comparator
        print("[2] Initializing ParallelComparator...")
        comparator = ParallelComparator(products_df, reviews_df)
        print()
        
        # Compare specific products
        print("[3] Comparing specific products (IDs: 1, 3, 5, 9)...")
        product_ids = [1, 3, 5, 9]
        results = comparator.compare_products(product_ids, n_workers=4)
        print()
        
        # Display results
        print("-" * 70)
        print("Product Comparison Results (sorted by avg_rating)")
        print("-" * 70)
        print(f"{'Rank':<6} {'Product':<25} {'Price':<10} {'Rating':<8} "
              f"{'Reviews':<8} {'Positive%':<10}")
        print("-" * 70)
        
        for i, product in enumerate(results, 1):
            print(f"{i:<6} {product['name']:<25} "
                  f"${product['price']:<9.2f} {product['avg_rating']:<8.2f} "
                  f"{product['review_count']:<8} "
                  f"{product['positive_ratio']*100:<9.1f}%")
        
        print()
        
        # Compare by category
        category_id = 1  # Electronics
        print(f"[4] Comparing all products in category {category_id} (Electronics)...")
        category_results = comparator.compare_by_category(category_id, n_workers=4)
        print()
        
        print("-" * 70)
        print(f"Category {category_id} Products (Top 5 by Rating)")
        print("-" * 70)
        print(f"{'Rank':<6} {'Product':<25} {'Price':<10} {'Rating':<8} "
              f"{'Reviews':<8}")
        print("-" * 70)
        
        for i, product in enumerate(category_results[:5], 1):
            print(f"{i:<6} {product['name']:<25} "
                  f"${product['price']:<9.2f} {product['avg_rating']:<8.2f} "
                  f"{product['review_count']:<8}")
        
        print()
        print("=" * 70)
        print("✓ ParallelComparator demonstration successful!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
