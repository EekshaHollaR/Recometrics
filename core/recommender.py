"""
Recommender Module
Parallel recommendation engine using collaborative filtering

This module provides the ParallelRecommender class that uses CPU-based
data parallelism via ProcessPoolExecutor to generate product recommendations
based on cosine similarity.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from core.device_manager import DeviceManager


def _cosine_similarity_chunk(args: tuple) -> np.ndarray:
    """
    Worker function to compute cosine similarity between matrix rows and target vector.
    
    This is a top-level function (required for ProcessPoolExecutor pickling).
    Computes cosine similarity: dot(A, B) / (||A|| * ||B||)
    
    Args:
        args: Tuple of (chunk_matrix, target_vector)
            - chunk_matrix: NumPy array of shape (n_samples, n_features)
            - target_vector: NumPy array of shape (n_features,)
    
    Returns:
        NumPy array of similarity scores, shape (n_samples,)
    
    Example:
        >>> chunk = np.array([[1, 0], [0, 1]])
        >>> target = np.array([1, 0])
        >>> scores = _cosine_similarity_chunk((chunk, target))
        >>> print(scores)
        [1.0, 0.0]
    """
    chunk_matrix, target_vector = args
    
    # Compute dot products: shape (n_samples,)
    dot_products = np.dot(chunk_matrix, target_vector)
    
    # Compute norms
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1)  # shape (n_samples,)
    target_norm = np.linalg.norm(target_vector)  # scalar
    
    # Avoid division by zero
    denominator = chunk_norms * target_norm
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    # Cosine similarity
    similarities = dot_products / denominator
    
    return similarities


class ParallelRecommender:
    """
    Parallel recommendation engine using cosine similarity.
    
    Uses CPU-based data parallelism via ProcessPoolExecutor to distribute
    similarity computations across multiple processes. Recommends products
    based on feature similarity.
    
    Attributes:
        dm (DeviceManager): Device manager for tensor operations
        product_matrix (np.ndarray | None): Product feature matrix (CPU)
        product_ids (np.ndarray | None): Product ID array
    
    Example:
        >>> from core.device_manager import DeviceManager
        >>> from core.features import FeatureBuilder
        >>> 
        >>> dm = DeviceManager()
        >>> recommender = ParallelRecommender(dm)
        >>> recommender.fit(features, product_ids)
        >>> recommendations = recommender.recommend_similar(0, top_k=5)
    """
    
    def __init__(self, device_manager: DeviceManager) -> None:
        """
        Initialize ParallelRecommender with a DeviceManager.
        
        Args:
            device_manager: DeviceManager instance for tensor operations
        
        Example:
            >>> dm = DeviceManager()
            >>> recommender = ParallelRecommender(dm)
        """
        self.dm = device_manager
        self.product_matrix: Optional[np.ndarray] = None
        self.product_ids: Optional[np.ndarray] = None
        
        print(f"[ParallelRecommender] Initialized with device: {device_manager.device}")
    
    def fit(self, product_matrix: Any, product_ids: np.ndarray) -> None:
        """
        Store product feature matrix and IDs for recommendation.
        
        Accepts either NumPy array or PyTorch tensor. Converts to NumPy
        array on CPU for ProcessPoolExecutor compatibility.
        
        Args:
            product_matrix: Feature matrix (np.ndarray or torch.Tensor)
                          Shape: (num_products, num_features)
            product_ids: Product ID array, shape (num_products,)
        
        Raises:
            ValueError: If shapes don't match or inputs are invalid
        
        Example:
            >>> features = np.array([[1, 2, 3], [4, 5, 6]])
            >>> ids = np.array([101, 102])
            >>> recommender.fit(features, ids)
        """
        # Convert torch tensor to NumPy if needed
        if hasattr(product_matrix, 'cpu'):
            # PyTorch tensor - convert to NumPy on CPU
            product_matrix_np = product_matrix.cpu().numpy()
            print("[ParallelRecommender] Converted torch tensor to NumPy array")
        elif isinstance(product_matrix, np.ndarray):
            product_matrix_np = product_matrix
        else:
            raise ValueError(
                f"product_matrix must be NumPy array or torch.Tensor, "
                f"got {type(product_matrix)}"
            )
        
        # Validate shapes
        if len(product_matrix_np.shape) != 2:
            raise ValueError(
                f"product_matrix must be 2D, got shape {product_matrix_np.shape}"
            )
        
        if len(product_ids.shape) != 1:
            raise ValueError(
                f"product_ids must be 1D, got shape {product_ids.shape}"
            )
        
        if product_matrix_np.shape[0] != product_ids.shape[0]:
            raise ValueError(
                f"Shape mismatch: product_matrix has {product_matrix_np.shape[0]} rows "
                f"but product_ids has {product_ids.shape[0]} elements"
            )
        
        # Store as NumPy arrays on CPU
        self.product_matrix = product_matrix_np.astype(np.float64)
        self.product_ids = product_ids.astype(np.int64)
        
        print(f"[ParallelRecommender] Fitted with {len(product_ids)} products, "
              f"{product_matrix_np.shape[1]} features")
    
    def recommend_similar(
        self,
        target_index: int,
        top_k: int = 10,
        n_workers: int = 4
    ) -> List[Dict[str, float]]:
        """
        Recommend similar products using parallel cosine similarity.
        
        Splits the product matrix into chunks and uses ProcessPoolExecutor
        to compute cosine similarity in parallel across CPU cores.
        
        Args:
            target_index: Index of the target product in product_matrix
            top_k: Number of recommendations to return (default: 10)
            n_workers: Number of parallel workers (default: 4)
        
        Returns:
            List of dictionaries with 'product_id' and 'score' keys,
            sorted by similarity score (highest first)
        
        Raises:
            ValueError: If not fitted or invalid parameters
        
        Example:
            >>> recommendations = recommender.recommend_similar(
            ...     target_index=0, top_k=5, n_workers=4
            ... )
            >>> for rec in recommendations:
            ...     print(f"Product {rec['product_id']}: {rec['score']:.3f}")
        """
        # Validate state
        if self.product_matrix is None or self.product_ids is None:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Validate target_index
        if target_index < 0 or target_index >= len(self.product_ids):
            raise ValueError(
                f"target_index {target_index} out of range [0, {len(self.product_ids)})"
            )
        
        # Validate top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        
        # Ensure top_k doesn't exceed available products (minus target)
        max_k = len(self.product_ids) - 1
        if top_k > max_k:
            top_k = max_k
            print(f"[ParallelRecommender] Warning: top_k adjusted to {max_k}")
        
        # Get target product vector
        target_vector = self.product_matrix[target_index]
        
        print(f"[ParallelRecommender] Computing similarities for product "
              f"{self.product_ids[target_index]} using {n_workers} workers...")
        
        # Split product matrix into chunks for parallel processing
        chunks = np.array_split(self.product_matrix, n_workers)
        
        # Prepare arguments for worker processes
        worker_args = [(chunk, target_vector) for chunk in chunks]
        
        # Execute parallel computation
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            partial_results = list(executor.map(_cosine_similarity_chunk, worker_args))
        
        # Concatenate partial similarity arrays
        similarities = np.concatenate(partial_results)
        
        # Exclude the target product itself (set similarity to -1)
        similarities[target_index] = -1.0
        
        # Find top_k indices using argpartition (efficient for large arrays)
        # Use negative similarities to get largest values
        if top_k < len(similarities):
            # argpartition: O(n) average case
            top_k_indices_unsorted = np.argpartition(-similarities, top_k)[:top_k]
        else:
            top_k_indices_unsorted = np.arange(len(similarities))
        
        # Sort top_k indices by similarity score (descending)
        top_k_scores = similarities[top_k_indices_unsorted]
        sorted_order = np.argsort(-top_k_scores)
        top_k_indices = top_k_indices_unsorted[sorted_order]
        
        # Build recommendation list
        recommendations = []
        for idx in top_k_indices:
            recommendations.append({
                'product_id': int(self.product_ids[idx]),
                'score': float(similarities[idx])
            })
        
        print(f"[ParallelRecommender] Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def __repr__(self) -> str:
        """
        String representation of ParallelRecommender.
        
        Returns:
            Formatted string showing configuration
        """
        fitted = self.product_matrix is not None
        n_products = len(self.product_ids) if fitted else 0
        
        return (
            f"ParallelRecommender(device='{self.dm.device}', "
            f"fitted={fitted}, n_products={n_products})"
        )


class ParallelRecommenderGPU(ParallelRecommender):
    """
    GPU-accelerated recommendation engine using vectorized PyTorch operations.
    
    Extends ParallelRecommender with GPU-based cosine similarity computation.
    Only available when DeviceManager detects CUDA-capable GPU and PyTorch.
    
    Uses fully vectorized operations on GPU for maximum performance with large datasets.
    Falls back to CPU-based parent class methods if GPU is unavailable.
    
    Attributes:
        dm (DeviceManager): Device manager (must have use_gpu=True)
        product_matrix (np.ndarray | None): Product feature matrix (CPU)
        product_ids (np.ndarray | None): Product ID array
    
    Example:
        >>> dm = DeviceManager()  # Detects GPU if available
        >>> if dm.use_gpu:
        ...     recommender = ParallelRecommenderGPU(dm)
        ...     recommender.fit(features, product_ids)
        ...     recs = recommender.recommend_similar_gpu(0, top_k=5)
    """
    
    def recommend_similar_gpu(
        self,
        target_index: int,
        top_k: int = 10
    ) -> List[Dict[str, float]]:
        """
        Recommend similar products using GPU-accelerated cosine similarity.
        
        Uses fully vectorized PyTorch operations on GPU for maximum performance.
        Computes cosine similarity for ALL products in a single GPU operation.
        
        Algorithm:
        1. Convert product_matrix to GPU tensor
        2. Extract target vector
        3. Compute: similarity = dot(M, target) / (||M|| * ||target||)
        4. Exclude target product
        5. Select top_k using torch.topk
        
        Args:
            target_index: Index of the target product in product_matrix
            top_k: Number of recommendations to return (default: 10)
        
        Returns:
            List of dictionaries with 'product_id' and 'score' keys,
            sorted by similarity score (highest first)
        
        Raises:
            RuntimeError: If GPU is not available or PyTorch not installed
            ValueError: If not fitted or invalid parameters
        
        Example:
            >>> recommendations = recommender.recommend_similar_gpu(
            ...     target_index=0, top_k=5
            ... )
            >>> for rec in recommendations:
            ...     print(f"Product {rec['product_id']}: {rec['score']:.4f}")
        """
        # Defensive checks for GPU availability
        if not self.dm.use_gpu:
            raise RuntimeError(
                "GPU not available. DeviceManager.use_gpu is False. "
                "Use recommend_similar() for CPU-based recommendations, "
                "or ensure CUDA is available."
            )
        
        if self.dm.torch is None:
            raise RuntimeError(
                "PyTorch not installed. Cannot use GPU recommendations. "
                "Install PyTorch with CUDA support or use recommend_similar() for CPU."
            )
        
        # Validate state
        if self.product_matrix is None or self.product_ids is None:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Validate target_index
        if target_index < 0 or target_index >= len(self.product_ids):
            raise ValueError(
                f"target_index {target_index} out of range [0, {len(self.product_ids)})"
            )
        
        # Validate top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        
        # Ensure top_k doesn't exceed available products (minus target)
        max_k = len(self.product_ids) - 1
        if top_k > max_k:
            top_k = max_k
            print(f"[ParallelRecommenderGPU] Warning: top_k adjusted to {max_k}")
        
        print(f"[ParallelRecommenderGPU] Computing similarities on GPU for product "
              f"{self.product_ids[target_index]}...")
        
        # Convert product matrix to GPU tensor
        torch = self.dm.torch
        M = torch.tensor(
            self.product_matrix,
            dtype=torch.float32,
            device=self.dm.device
        )
        
        # Extract target vector (shape: [n_features])
        target_vec = M[target_index]
        
        # Compute cosine similarity in a fully vectorized way
        # dot products: M @ target_vec -> shape [n_products]
        dot = torch.matmul(M, target_vec)
        
        # Compute norms
        norm_M = torch.norm(M, dim=1)  # shape [n_products]
        norm_v = torch.norm(target_vec)  # scalar
        
        # Cosine similarity: dot / (||M|| * ||v||)
        similarities = dot / (norm_M * norm_v + 1e-9)
        
        # Exclude the target product (set to very low value)
        similarities[target_index] = -1.0
        
        # Find top_k indices using torch.topk (returns sorted results)
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k, largest=True)
        
        # Convert to CPU and NumPy for output
        top_k_indices_np = top_k_indices.cpu().numpy()
        top_k_values_np = top_k_values.cpu().numpy()
        
        # Build recommendation list
        recommendations = []
        for i in range(len(top_k_indices_np)):
            idx = top_k_indices_np[i]
            recommendations.append({
                'product_id': int(self.product_ids[idx]),
                'score': float(top_k_values_np[i])
            })
        
        print(f"[ParallelRecommenderGPU] Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def __repr__(self) -> str:
        """
        String representation of ParallelRecommenderGPU.
        
        Returns:
            Formatted string showing configuration
        """
        fitted = self.product_matrix is not None
        n_products = len(self.product_ids) if fitted else 0
        
        return (
            f"ParallelRecommenderGPU(device='{self.dm.device}', "
            f"use_gpu={self.dm.use_gpu}, fitted={fitted}, n_products={n_products})"
        )


if __name__ == "__main__":
    """
    Demonstration of ParallelRecommender functionality.
    
    Run this script to test the recommendation engine:
        python core/recommender.py
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from core.device_manager import DeviceManager
    from core.data_loader import load_products
    from core.features import FeatureBuilder
    
    print("=" * 70)
    print("PaReCo-Py ParallelRecommender Demo")
    print("=" * 70)
    
    try:
        # Initialize DeviceManager
        print("\n[1] Initializing DeviceManager...")
        dm = DeviceManager()
        print()
        
        # Load products
        print("[2] Loading product data...")
        data_dir = current_dir.parent / "data"
        products_path = data_dir / "products.csv"
        products_df = load_products(str(products_path))
        print()
        
        # Build features
        print("[3] Building feature matrix...")
        fb = FeatureBuilder(dm)
        features = fb.build_product_matrix(products_df)
        product_ids = fb.get_product_ids(products_df)
        print()
        
        # Initialize recommender
        print("[4] Initializing ParallelRecommender...")
        recommender = ParallelRecommender(dm)
        print()
        
        # Fit recommender
        print("[5] Fitting recommender with product data...")
        recommender.fit(features, product_ids)
        print()
        
        # Generate recommendations
        target_idx = 0
        target_product_id = product_ids[target_idx]
        target_name = products_df.iloc[target_idx]['name']
        
        print(f"[6] Generating recommendations for product {target_product_id}: '{target_name}'")
        recommendations = recommender.recommend_similar(
            target_index=target_idx,
            top_k=5,
            n_workers=4
        )
        print()
        
        # Display results
        print("-" * 70)
        print(f"Top 5 Recommendations for '{target_name}' (Product {target_product_id})")
        print("-" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            # Find product info
            product_row = products_df[products_df['product_id'] == rec['product_id']].iloc[0]
            print(f"{i}. Product {rec['product_id']:2d}: {product_row['name']:25s} "
                  f"(Similarity: {rec['score']:.4f})")
        
        print()
        
        # Test with another product
        target_idx_2 = 4  # Mechanical Keyboard
        target_product_id_2 = product_ids[target_idx_2]
        target_name_2 = products_df.iloc[target_idx_2]['name']
        
        print("-" * 70)
        print(f"Top 5 Recommendations for '{target_name_2}' (Product {target_product_id_2})")
        print("-" * 70)
        
        recommendations_2 = recommender.recommend_similar(
            target_index=target_idx_2,
            top_k=5,
            n_workers=4
        )
        
        for i, rec in enumerate(recommendations_2, 1):
            product_row = products_df[products_df['product_id'] == rec['product_id']].iloc[0]
            print(f"{i}. Product {rec['product_id']:2d}: {product_row['name']:25s} "
                  f"(Similarity: {rec['score']:.4f})")
        
        print()
        print("=" * 70)
        print("✓ ParallelRecommender demonstration successful!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
