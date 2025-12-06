"""
Streamlit Web Application
Interactive UI for PaReCo-Py recommendation and comparison engine

Run: streamlit run api/app_streamlit.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from core.device_manager import DeviceManager
from core.data_loader import load_products, load_reviews, validate_data
from core.features import FeatureBuilder
from core.recommender import ParallelRecommender, ParallelRecommenderGPU
from core.comparator import ParallelComparator
from core.metrics import time_function, format_time


# ========== Page Configuration ==========
st.set_page_config(
    page_title="PaReCo-Py",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== Initialize Components (Cached) ==========
@st.cache_resource
def initialize_system():
    """Initialize all system components (runs once on startup)."""
    
    # Device manager
    dm = DeviceManager()
    
    # Load data
    data_dir = current_dir.parent / "data"
    products = load_products(str(data_dir / "products.csv"))
    reviews = load_reviews(str(data_dir / "reviews.csv"))
    validate_data(products, reviews)
    
    # Build features
    fb = FeatureBuilder(dm)
    features = fb.build_product_matrix(products)
    product_ids = fb.get_product_ids(products)
    
    # Create recommenders
    recommender_cpu = ParallelRecommender(dm)
    recommender_cpu.fit(features, product_ids)
    
    recommender_gpu = None
    if dm.use_gpu:
        recommender_gpu = ParallelRecommenderGPU(dm)
        recommender_gpu.fit(features, product_ids)
    
    # Create comparator
    comparator = ParallelComparator(products, reviews)
    
    return {
        'dm': dm,
        'products': products,
        'reviews': reviews,
        'product_ids': product_ids,
        'recommender_cpu': recommender_cpu,
        'recommender_gpu': recommender_gpu,
        'comparator': comparator
    }


# Initialize system
system = initialize_system()

dm = system['dm']
products = system['products']
reviews = system['reviews']
product_ids = system['product_ids']
recommender_cpu = system['recommender_cpu']
recommender_gpu = system['recommender_gpu']
comparator = system['comparator']


# ========== Sidebar ==========
with st.sidebar:
    st.title("🛍️ PaReCo-Py")
    st.markdown("**Parallel Recommendation & Comparison Engine**")
    
    st.divider()
    
    st.subheader("⚙️ System Information")
    
    # Device info
    if dm.use_gpu:
        st.success(f"**Device:** GPU ({dm.device})")
        st.info("✓ CUDA acceleration enabled")
    else:
        st.info(f"**Device:** CPU only")
        st.caption("💡 Install PyTorch with CUDA for GPU acceleration")
    
    st.divider()
    
    # Data statistics
    st.subheader("📊 Dataset")
    st.metric("Products", len(products))
    st.metric("Reviews", len(reviews))
    st.metric("Categories", products['category_id'].nunique())
    
    st.divider()
    
    # About
    st.caption("**About PaReCo-Py**")
    st.caption("Demonstrates three parallelism patterns:")
    st.caption("• Data Parallelism (CPU)")
    st.caption("• Task Parallelism (CPU)")
    if dm.use_gpu:
        st.caption("• GPU Parallelism (PyTorch)")


# ========== Main Content ==========
st.title("🛍️ PaReCo-Py Dashboard")
st.markdown("Parallel recommendation and comparison system with real-time performance metrics")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Recommendations", "⚖️ Product Comparison"])


# ========== Tab 1: Recommendations ==========
with tab1:
    st.header("Product Recommendations")
    st.markdown("Get similar product recommendations using cosine similarity")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Product selection
        product_names = products['name'].tolist()
        selected_product_name = st.selectbox(
            "Select a product:",
            product_names,
            index=0
        )
    
    with col2:
        # Top-k slider
        top_k = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=min(20, len(products) - 1),
            value=5
        )
    
    # Mode selection
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if dm.use_gpu:
            mode = st.radio(
                "Computation mode:",
                ["CPU Parallel (4 workers)", "GPU Accelerated"],
                index=1
            )
        else:
            mode = st.radio(
                "Computation mode:",
                ["CPU Parallel (4 workers)"],
                index=0
            )
            st.caption("⚠️ GPU not available")
    
    # Get recommendations button
    if st.button("🚀 Get Recommendations", type="primary"):
        # Find product index
        target_index = products[products['name'] == selected_product_name].index[0]
        target_product = products.iloc[target_index]
        
        # Display selected product info
        st.subheader("Selected Product")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        col_info1.metric("Product ID", target_product['product_id'])
        col_info2.metric("Price", f"${target_product['price']:.2f}")
        col_info3.metric("Rating", f"{target_product['avg_rating']:.2f}⭐")
        col_info4.metric("Category", target_product['category_id'])
        
        st.divider()
        
        # Generate recommendations
        with st.spinner("Computing similarities..."):
            if mode == "GPU Accelerated" and recommender_gpu:
                recs, elapsed = time_function(
                    recommender_gpu.recommend_similar_gpu,
                    target_index,
                    top_k=top_k
                )
                mode_label = "GPU"
            else:
                recs, elapsed = time_function(
                    recommender_cpu.recommend_similar,
                    target_index,
                    top_k=top_k,
                    n_workers=4
                )
                mode_label = "CPU (4 workers)"
        
        # Display results
        st.success(f"✓ Found {len(recs)} recommendations in {format_time(elapsed)} using {mode_label}")
        
        # Build results dataframe
        results_data = []
        for i, rec in enumerate(recs, 1):
            prod = products[products['product_id'] == rec['product_id']].iloc[0]
            results_data.append({
                'Rank': i,
                'Product ID': rec['product_id'],
                'Product Name': prod['name'],
                'Price': f"${prod['price']:.2f}",
                'Rating': f"{prod['avg_rating']:.2f}⭐",
                'Similarity': f"{rec['score']:.6f}"
            })
        
        results_df = pd.DataFrame(results_data)
        
        st.subheader("Top Recommendations")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Performance metrics
        with st.expander("📊 Performance Metrics"):
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            perf_col1.metric("Computation Time", format_time(elapsed))
            perf_col2.metric("Mode", mode_label)
            perf_col3.metric("Products Analyzed", len(products))


# ========== Tab 2: Comparison ==========
with tab2:
    st.header("Product Comparison")
    st.markdown("Compare multiple products side-by-side with review analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-select for products
        selected_products = st.multiselect(
            "Select products to compare (2-6 products):",
            product_names,
            default=product_names[:3]
        )
    
    with col2:
        # Workers slider
        n_workers = st.slider(
            "Number of workers:",
            min_value=1,
            max_value=8,
            value=4
        )
    
    # Validate selection
    if len(selected_products) < 2:
        st.warning("⚠️ Please select at least 2 products to compare")
    elif len(selected_products) > 6:
        st.warning("⚠️ Please select at most 6 products for optimal display")
    
    # Compare button
    if st.button("⚖️ Compare Products", type="primary", disabled=len(selected_products) < 2):
        # Resolve names to product IDs
        comparison_ids = []
        for name in selected_products:
            pid = products[products['name'] == name]['product_id'].iloc[0]
            comparison_ids.append(pid)
        
        # Perform comparison
        with st.spinner(f"Analyzing {len(comparison_ids)} products with {n_workers} workers..."):
            comparison_results, elapsed = time_function(
                comparator.compare_products,
                comparison_ids,
                n_workers=n_workers
            )
        
        st.success(f"✓ Compared {len(comparison_results)} products in {format_time(elapsed)}")
        
        # Build comparison dataframe
        comparison_data = []
        for i, prod in enumerate(comparison_results, 1):
            comparison_data.append({
                'Rank': i,
                'Product Name': prod['name'],
                'Price': f"${prod['price']:.2f}",
                'Avg Rating': f"{prod['avg_rating']:.2f}⭐",
                'Review Count': prod['review_count'],
                'Positive Reviews': f"{prod['positive_ratio']*100:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.subheader("Comparison Results (Sorted by Rating)")
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visual comparison
        st.subheader("📊 Price vs Rating")
        
        # Create chart data
        chart_data = pd.DataFrame({
            'Product': [p['name'] for p in comparison_results],
            'Price': [p['price'] for p in comparison_results],
            'Rating': [p['avg_rating'] for p in comparison_results]
        })
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.bar_chart(chart_data.set_index('Product')['Price'])
        
        with col_chart2:
            st.bar_chart(chart_data.set_index('Product')['Rating'])
        
        # Performance metrics
        with st.expander("📊 Performance Metrics"):
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            perf_col1.metric("Comparison Time", format_time(elapsed))
            perf_col2.metric("Workers", n_workers)
            perf_col3.metric("Products Analyzed", len(comparison_results))


# ========== Footer ==========
st.divider()
st.caption("PaReCo-Py - Parallel Recommendation & Comparison Engine | "
          "Demonstrates data parallelism, task parallelism, and GPU acceleration")
