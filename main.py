"""
PaReCo-Py Main CLI Demo Runner
Parallel Recommendation & Comparison Engine - Complete Demonstration

This script demonstrates all core modules:
- DeviceManager: Hardware detection
- DataLoader: CSV loading and validation
- FeatureBuilder: Feature extraction
- ParallelRecommender: CPU-based recommendations
- ParallelRecommenderGPU: GPU-accelerated recommendations (if available)
- ParallelComparator: Product comparison
- Metrics: Performance measurement

Run: python main.py
"""

from pathlib import Path
from core.device_manager import DeviceManager
from core.data_loader import load_products, load_reviews, validate_data
from core.features import FeatureBuilder
from core.recommender import ParallelRecommender, ParallelRecommenderGPU
from core.comparator import ParallelComparator
from core.metrics import time_function, compute_speedup, compute_efficiency, format_time


def print_banner(title: str, width: int = 70) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_section(title: str, width: int = 70) -> None:
    """Print a section header."""
    print("\n" + "-" * width)
    print(f"[{title}]")
    print("-" * width)


def main():
    """Main CLI demo execution."""
    
    print_banner("🛍️  PaReCo-Py - Parallel Recommendation & Comparison Engine")
    print("Complete system demonstration with performance benchmarking")
    
    # ========== Step 1: Initialize DeviceManager ==========
    print_section("Step 1: Hardware Detection")
    dm = DeviceManager()
    print(f"  • Device: {dm.device}")
    print(f"  • GPU Available: {dm.use_gpu}")
    print(f"  • PyTorch Installed: {dm.torch is not None}")
    
    # ========== Step 2: Load Data ==========
    print_section("Step 2: Loading Data")
    
    data_dir = Path(__file__).parent / "data"
    products_path = data_dir / "products.csv"
    reviews_path = data_dir / "reviews.csv"
    
    print(f"  • Loading products from: {products_path.name}")
    products = load_products(str(products_path))
    
    print(f"  • Loading reviews from: {reviews_path.name}")
    reviews = load_reviews(str(reviews_path))
    
    print(f"  • Validating data integrity...")
    validate_data(products, reviews)
    print(f"  ✓ Data validation passed")
    
    # ========== Step 3: Build Features ==========
    print_section("Step 3: Feature Engineering")
    
    fb = FeatureBuilder(dm)
    
    print(f"  • Building feature matrix...")
    features, t_features = time_function(fb.build_product_matrix, products)
    product_ids = fb.get_product_ids(products)
    
    print(f"  • Feature matrix shape: {features.shape}")
    print(f"  • Feature building time: {format_time(t_features)}")
    
    # ========== Step 4: CPU Recommendations ==========
    print_section("Step 4: CPU-Based Parallel Recommendations")
    
    recommender_cpu = ParallelRecommender(dm)
    recommender_cpu.fit(features, product_ids)
    
    target_index = 0
    target_product_id = product_ids[target_index]
    target_product = products.iloc[target_index]
    
    print(f"  • Target Product: [{target_product_id}] {target_product['name']}")
    print(f"  • Price: ${target_product['price']:.2f}")
    print(f"  • Category: {target_product['category_id']}")
    print(f"\n  • Generating recommendations with 4 CPU workers...")
    
    recs_cpu, t_cpu = time_function(
        recommender_cpu.recommend_similar,
        target_index,
        top_k=5,
        n_workers=4
    )
    
    print(f"  • CPU Recommendation time: {format_time(t_cpu)}")
    print(f"\n  Top 5 Similar Products (CPU):")
    print(f"  {'Rank':<6} {'ID':<6} {'Product':<30} {'Similarity':<12}")
    print(f"  {'-'*60}")
    
    for i, rec in enumerate(recs_cpu, 1):
        prod = products[products['product_id'] == rec['product_id']].iloc[0]
        print(f"  {i:<6} {rec['product_id']:<6} {prod['name']:<30} {rec['score']:.6f}")
    
    # ========== Step 5: GPU Recommendations (if available) ==========
    if dm.use_gpu:
        print_section("Step 5: GPU-Accelerated Recommendations")
        
        recommender_gpu = ParallelRecommenderGPU(dm)
        recommender_gpu.fit(features, product_ids)
        
        print(f"  • Generating recommendations on GPU...")
        
        recs_gpu, t_gpu = time_function(
            recommender_gpu.recommend_similar_gpu,
            target_index,
            top_k=5
        )
        
        print(f"  • GPU Recommendation time: {format_time(t_gpu)}")
        print(f"\n  Top 5 Similar Products (GPU):")
        print(f"  {'Rank':<6} {'ID':<6} {'Product':<30} {'Similarity':<12}")
        print(f"  {'-'*60}")
        
        for i, rec in enumerate(recs_gpu, 1):
            prod = products[products['product_id'] == rec['product_id']].iloc[0]
            print(f"  {i:<6} {rec['product_id']:<6} {prod['name']:<30} {rec['score']:.6f}")
        
        # Performance comparison
        print(f"\n  Performance Comparison:")
        speedup = compute_speedup(t_cpu, t_gpu)
        print(f"  • CPU Time:  {format_time(t_cpu)}")
        print(f"  • GPU Time:  {format_time(t_gpu)}")
        print(f"  • Speedup:   {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✓ GPU is {speedup:.2f}x faster than CPU")
        else:
            print(f"  ⚠ GPU slower for this small dataset (overhead dominates)")
    
    else:
        print_section("Step 5: GPU Recommendations")
        print(f"  ⚠ GPU not available - skipping GPU benchmark")
        print(f"  • To enable GPU: Install PyTorch with CUDA support")
    
    # ========== Step 6: Product Comparison ==========
    print_section("Step 6: Parallel Product Comparison")
    
    comparator = ParallelComparator(products, reviews)
    
    # Compare the recommended products
    comparison_ids = [rec['product_id'] for rec in recs_cpu[:4]]
    
    print(f"  • Comparing {len(comparison_ids)} products using 4 workers...")
    print(f"  • Product IDs: {comparison_ids}")
    
    comparison, t_comparison = time_function(
        comparator.compare_products,
        comparison_ids,
        n_workers=4
    )
    
    print(f"  • Comparison time: {format_time(t_comparison)}")
    
    print(f"\n  Product Comparison Table (sorted by rating):")
    print(f"  {'Rank':<6} {'Product':<25} {'Price':<10} {'Rating':<8} {'Reviews':<8} {'Positive':<10}")
    print(f"  {'-'*70}")
    
    for i, prod in enumerate(comparison, 1):
        print(f"  {i:<6} {prod['name']:<25} "
              f"${prod['price']:<9.2f} {prod['avg_rating']:<8.2f} "
              f"{prod['review_count']:<8} {prod['positive_ratio']*100:<9.1f}%")
    
    # ========== Step 7: Performance Summary ==========
    print_section("Step 7: Overall Performance Summary")
    
    total_time = t_features + t_cpu + t_comparison
    if dm.use_gpu:
        total_time += t_gpu
    
    print(f"  Component Breakdown:")
    print(f"  • Feature Building:      {format_time(t_features):>10}")
    print(f"  • CPU Recommendation:    {format_time(t_cpu):>10}")
    if dm.use_gpu:
        print(f"  • GPU Recommendation:    {format_time(t_gpu):>10}")
    print(f"  • Product Comparison:    {format_time(t_comparison):>10}")
    print(f"  " + "-" * 40)
    print(f"  • Total Pipeline Time:   {format_time(total_time):>10}")
    
    # Efficiency analysis
    print(f"\n  Parallel Efficiency Analysis:")
    
    # CPU recommender efficiency (assuming sequential would be ~4x slower)
    cpu_workers = 4
    # Rough estimate: sequential ≈ parallel * workers (if perfect)
    # In practice, we'd need to measure 1 worker for accurate comparison
    print(f"  • CPU Workers: {cpu_workers}")
    print(f"  • CPU Recommendation: {format_time(t_cpu)}")
    print(f"  • Note: Run with n_workers=1 for accurate speedup measurement")
    
    # Final banner
    print_banner("✓ Demonstration Complete!")
    
    print("\nKey Takeaways:")
    print("  1. ✓ Hardware detection works (CPU/GPU)")
    print("  2. ✓ Data loading and validation successful")
    print("  3. ✓ Feature engineering with standardization")
    print("  4. ✓ CPU parallel recommendations using ProcessPoolExecutor")
    if dm.use_gpu:
        print("  5. ✓ GPU acceleration using PyTorch")
    else:
        print("  5. ⚠ GPU not available (CPU-only mode)")
    print("  6. ✓ Product comparison with ThreadPoolExecutor")
    print("  7. ✓ Performance measurement and benchmarking")
    
    print("\nPaReCo-Py demonstrates three parallelism patterns:")
    print("  • Data Parallelism (CPU): Split data across processes")
    print("  • Task Parallelism (CPU): Independent tasks across threads")
    if dm.use_gpu:
        print("  • GPU Parallelism: Vectorized operations on CUDA cores")
    
    print("\n" + "=" * 70)
    print("Thank you for using PaReCo-Py!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
