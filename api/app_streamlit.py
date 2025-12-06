"""
Streamlit Web Application
Interactive UI for PaReCo-Py recommendation and comparison engine

Optimized for large datasets (50K+ products) with:
- Search-based product selection (instead of dropdowns)
- Product image display
- Efficient rendering (only displays filtered results)

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


# ========== Utility Functions ==========
def search_products(
    products_df: pd.DataFrame,
    query: str,
    max_results: int = 20
) -> pd.DataFrame:
    """
    Search products by name (optimized for 50K+ rows).
    
    Uses vectorized string operations for efficiency. Case-insensitive search
    on product names, sorted by match position (earlier matches first).
    
    Args:
        products_df: DataFrame with product data
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        Filtered DataFrame with up to max_results products
    
    Example:
        >>> results = search_products(products, "wireless", max_results=10)
    """
    if not query or query.strip() == "":
        return pd.DataFrame()  # Empty result for empty query
    
    query_lower = query.lower().strip()
    
    # Vectorized case-insensitive search
    matches = products_df['name'].str.lower().str.contains(query_lower, na=False)
    filtered = products_df[matches].copy()
    
    if filtered.empty:
        return filtered
    
    # Sort by match position (earlier matches ranked higher)
    filtered['_match_pos'] = filtered['name'].str.lower().apply(
        lambda x: x.find(query_lower)
    )
    filtered = filtered.sort_values('_match_pos')
    filtered = filtered.drop(columns=['_match_pos'])
    
    # Return top results
    return filtered.head(max_results)


def get_product_details(products_df: pd.DataFrame, product_id: int) -> dict:
    """
    Get detailed information for a specific product.
    
    Args:
        products_df: DataFrame with product data
        product_id: Product ID to look up
    
    Returns:
        Dictionary with product details: product_id, name, price, avg_rating, category_id, image_url
    
    Raises:
        KeyError: If product_id not found
    
    Example:
        >>> details = get_product_details(products, 42)
        >>> print(f"{details['name']}: ${details['price']:.2f}")
    """
    product_row = products_df[products_df['product_id'] == product_id]
    
    if product_row.empty:
        raise KeyError(f"Product ID {product_id} not found in products DataFrame")
    
    row = product_row.iloc[0]
    
    return {
        'product_id': int(row['product_id']),
        'name': str(row['name']),
        'price': float(row['price']),
        'avg_rating': float(row['avg_rating']),
        'category_id': int(row['category_id']),
        'image_url': str(row.get('image_url', 'NO_IMAGE'))
    }


def display_product_image(image_url: str, width: int = 150):
    """
    Display product image or placeholder.
    
    Args:
        image_url: URL or path to image
        width: Display width in pixels
    """
    if image_url and image_url != "NO_IMAGE":
        try:
            st.image(image_url, width=width)
        except Exception:
            st.caption("📷 Image unavailable")
    else:
        st.caption("📷 No image available")


# ========== Initialize Components (Cached) ==========
@st.cache_data
def get_data():
    """
    Load product and review data (cached for performance).
    
    Uses @st.cache_data so that CSV loading happens only once,
    even with 50K+ products. Subsequent interactions reuse cached data.
    
    Returns:
        Tuple of (products_df, reviews_df)
    """
    data_dir = current_dir.parent / "data"
    products = load_products(str(data_dir / "products.csv"))
    reviews = load_reviews(str(data_dir / "reviews.csv"))
    validate_data(products, reviews)
    return products, reviews


@st.cache_resource
def get_models(products_df: pd.DataFrame, reviews_df: pd.DataFrame):
    """
    Initialize recommenders and comparator (cached as resources).
    
    Uses @st.cache_resource for expensive operations:
    - DeviceManager initialization
    - Feature matrix building (float32 operations)
    - Model fitting (storing 50K+ product vectors)
    
    These are resource-heavy and should only happen once per session.
    
    Args:
        products_df: Products DataFrame
        reviews_df: Reviews DataFrame
    
    Returns:
        Dictionary with dm, product_ids, recommender_cpu, recommender_gpu, comparator
    """
    # Device manager
    dm = DeviceManager()
    
    # Build features (expensive for 50K+ products)
    fb = FeatureBuilder(dm)
    features = fb.build_product_matrix(products_df)
    product_ids = fb.get_product_ids(products_df)
    
    # Create recommenders (stores feature matrices)
    recommender_cpu = ParallelRecommender(dm)
    recommender_cpu.fit(features, product_ids)
    
    recommender_gpu = None
    if dm.use_gpu:
        recommender_gpu = ParallelRecommenderGPU(dm)
        recommender_gpu.fit(features, product_ids)
    
    # Create comparator
    comparator = ParallelComparator(products_df, reviews_df)
    
    return {
        'dm': dm,
        'product_ids': product_ids,
        'recommender_cpu': recommender_cpu,
        'recommender_gpu': recommender_gpu,
        'comparator': comparator
    }


# Load data (cached)
products, reviews = get_data()

# Initialize models (cached)
models = get_models(products, reviews)
dm = models['dm']
product_ids = models['product_ids']
recommender_cpu = models['recommender_cpu']
recommender_gpu = models['recommender_gpu']
comparator = models['comparator']


# ========== Sidebar ==========
with st.sidebar:
    st.title("🛍️ PaReCo-Py")
    st.markdown("**Parallel Recommendation & Comparison Engine**")
    st.caption("Optimized for 50,000+ products")
    
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
    st.caption("• Search-based selection for 50K+ products")
    st.caption("• Product image display")
    st.caption("• Data Parallelism (CPU)")
    st.caption("• Task Parallelism (CPU)")
    if dm.use_gpu:
        st.caption("• GPU Parallelism (PyTorch)")


# ========== Main Content ==========
st.title("🛍️ PaReCo-Py Dashboard")
st.markdown("Search-based product discovery for large catalogs (50K+ products)")

# Optimization info
st.info(
    "🚀 **Optimized for Large Datasets**: This application uses Streamlit caching "
    "and parallel processing to efficiently handle 50,000+ products. Data loading "
    "and model initialization happen only once per session, ensuring fast interactions."
)

# Create tabs
tab1, tab2 = st.tabs(["🎯 Recommendations", "⚖️ Product Comparison"])


# ========== Tab 1: Recommendations ==========
with tab1:
    st.header("Product Recommendations")
    st.markdown("Search for a product, view details, and get personalized recommendations")
    
    # Initialize session state for recommendations  
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'rec_limit' not in st.session_state:
        st.session_state.rec_limit = 6
    if 'selected_product_id' not in st.session_state:
        st.session_state.selected_product_id = None
    
    # Search interface
    search_query = st.text_input(
        "🔍 Search for a product by name:",
        placeholder="e.g., wireless, headphones, keyboard..."
    )
    
    selected_product_id = None
    
    if search_query:
        # Search products (up to 30 results)
        search_results = search_products(products, search_query, max_results=30)
        
        if not search_results.empty:
            st.success(f"Found {len(search_results)} matching products (showing top 30)")
            
            # Product selection from search results with radio buttons
            product_options = [
                f"{row['product_id']} – {row['name']}"
                for _, row in search_results.iterrows()
            ]
            
            selected_option = st.radio(
                "Select a product:",
                options=product_options,
                index=0,
                key="product_selector"
            )
            
            # Extract product ID from selection
            selected_product_id = int(selected_option.split(" – ")[0])
            
            # If product changed, reset recommendations
            if selected_product_id != st.session_state.selected_product_id:
                st.session_state.selected_product_id = selected_product_id
                st.session_state.recommendations = None
                st.session_state.rec_limit = 6
            
        else:
            st.warning("❌ No products found. Try a different search term.")
    else:
        st.info("👆 Start typing to search for products")
    
    # ========== Product Details Section ==========
    if selected_product_id:
        st.divider()
        st.subheader("📦 Product Details")
        
        # Get product details
        target_product = get_product_details(products, selected_product_id)
        
        # Layout: Image left, Details right
        col_img, col_details = st.columns([1, 2])
        
        with col_img:
            display_product_image(target_product['image_url'], width=200)
        
        with col_details:
            st.markdown(f"### {target_product['name']}")
            
            # Product metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Price", f"${target_product['price']:.2f}")
            metric_col2.metric("Avg Rating", f"{target_product['avg_rating']:.1f}⭐")
            metric_col3.metric("Category ID", target_product['category_id'])
            
            # Auto-generate description if missing
            category_names = {
                1: "Electronics", 2: "Accessories", 3: "Computing", 
                4: "Audio", 5: "Peripherals", 6: "Cables & Adapters"
            }
            category_id = target_product['category_id']
            category_name = category_names.get(category_id, "Products")
            
            # Description (auto-generated)
            st.markdown("**Description:**")
            description = (
                f"This {target_product['name']} is a high-quality product in our {category_name} category. "
                f"Priced at ${target_product['price']:.2f}, it has earned an average rating of "
                f"{target_product['avg_rating']:.1f} stars from our customers. "
                f"Perfect for anyone looking for reliable {category_name.lower()}."
            )
            st.write(description)
        
        # ========== Reviews Section ==========
        st.markdown("---")
        st.markdown("### 📝 Recent Reviews")
        
        # Filter reviews for this product
        product_reviews = reviews[reviews['product_id'] == selected_product_id].head(20)
        
        if not product_reviews.empty:
            # Display reviews in a clean format
            for idx, review_row in product_reviews.iterrows():
                review_col1, review_col2 = st.columns([1, 5])
                
                with review_col1:
                    st.metric("Rating", f"{review_row['rating']:.1f}⭐", label_visibility="collapsed")
                
                with review_col2:
                    st.markdown(f"*{review_row['review_text']}*")
                
                if idx < product_reviews.index[-1]:
                    st.markdown("---")
        else:
            st.info("📭 No reviews available for this product yet.")
        
        # ========== Auto-Generated Recommendations ==========
        st.divider()
        st.subheader("🎯 Recommended Products")
        
        # Auto-compute recommendations if not already done
        if st.session_state.recommendations is None:
            target_index = products[products['product_id'] == selected_product_id].index[0]
            
            with st.spinner("Computing personalized recommendations..."):
                # Always compute top 20 recommendations
                if dm.use_gpu and recommender_gpu:
                    recs, elapsed = time_function(
                        recommender_gpu.recommend_similar_gpu,
                        target_index,
                        top_k=20
                    )
                    mode_label = "GPU"
                else:
                    recs, elapsed = time_function(
                        recommender_cpu.recommend_similar,
                        target_index,
                        top_k=20,
                        n_workers=4
                    )
                    mode_label = "CPU (4 workers)"
                
                st.session_state.recommendations = recs
                st.session_state.computation_time = elapsed
                st.session_state.computation_mode = mode_label
        
        # Display recommendations (progressive reveal)
        if st.session_state.recommendations:
            recs = st.session_state.recommendations
            rec_limit = st.session_state.rec_limit
            
            st.info(f"✨ Showing {min(rec_limit, len(recs))} of {len(recs)} recommendations "
                   f"(computed in {format_time(st.session_state.computation_time)} "
                   f"using {st.session_state.computation_mode})")
            
            # Display recommendations up to rec_limit
            for i, rec in enumerate(recs[:rec_limit], 1):
                prod_details = get_product_details(products, rec['product_id'])
                
                # Card-like layout with columns
                card_col1, card_col2 = st.columns([1, 3])
                
                with card_col1:
                    display_product_image(prod_details['image_url'], width=120)
                
                with card_col2:
                    st.markdown(f"#### {i}. {prod_details['name']}")
                    
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    detail_col1.metric("Price", f"${prod_details['price']:.2f}")
                    detail_col2.metric("Rating", f"{prod_details['avg_rating']:.1f}⭐")
                    detail_col3.metric("Similarity", f"{rec['score']:.4f}")
                
                if i < min(rec_limit, len(recs)):
                    st.markdown("---")
            
            # "Load More" button
            if rec_limit < len(recs):
                if st.button("📥 Load More Recommendations", type="secondary"):
                    st.session_state.rec_limit = min(rec_limit + 6, 20)
                    st.rerun()
            elif rec_limit < 20 and len(recs) == rec_limit:
                st.caption("✓ All recommendations loaded")
            
            # Performance metrics
            with st.expander("📊 Performance Metrics"):
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("Computation Time", format_time(st.session_state.computation_time))
                perf_col2.metric("Mode", st.session_state.computation_mode)
                perf_col3.metric("Products Analyzed", len(products))


# ========== Tab 2: Comparison ==========
with tab2:
    st.header("Product Comparison")
    st.markdown("Search and compare multiple products side-by-side")
    
    # Search interface
    comp_search_query = st.text_input(
        "🔍 Search for products to compare:",
        placeholder="e.g., mouse, cable, charger...",
        key="comp_search"
    )
    
    selected_for_comparison = []
    
    if comp_search_query:
        # Search products
        comp_search_results = search_products(products, comp_search_query, max_results=20)
        
        if not comp_search_results.empty:
            st.success(f"Found {len(comp_search_results)} matching products")
            
            # Display search results
            display_comp_results = comp_search_results[['product_id', 'name', 'price', 'avg_rating']].copy()
            display_comp_results['price'] = display_comp_results['price'].apply(lambda x: f"${x:.2f}")
            display_comp_results['avg_rating'] = display_comp_results['avg_rating'].apply(lambda x: f"{x:.1f}⭐")
            
            st.dataframe(
                display_comp_results,
                use_container_width=True,
                hide_index=True
            )
            
            # Multi-select from search results
            comp_options = [
                f"{row['product_id']} – {row['name']}"
                for _, row in comp_search_results.iterrows()
            ]
            
            selected_comp_options = st.multiselect(
                "Select 2-6 products to compare:",
                options=comp_options,
                default=comp_options[:min(3, len(comp_options))]
            )
            
            # Extract product IDs
            selected_for_comparison = [
                int(opt.split(" – ")[0]) for opt in selected_comp_options
            ]
            
        else:
            st.warning("❌ No products found. Try a different search term.")
    else:
        st.info("👆 Start typing to search for products to compare")
    
    # Workers slider
    n_workers = st.slider(
        "Number of workers:",
        min_value=1,
        max_value=8,
        value=4
    )
    
    # Validation
    if selected_for_comparison and (len(selected_for_comparison) < 2 or len(selected_for_comparison) > 6):
        st.warning("⚠️ Please select between 2 and 6 products for comparison")
    
    # Compare button
    if st.button(
        "⚖️ Compare Products",
        type="primary",
        disabled=(len(selected_for_comparison) < 2 or len(selected_for_comparison) > 6)
    ):
        # Display selected products with images
        st.subheader("Selected Products")
        cols = st.columns(min(len(selected_for_comparison), 3))
        
        for idx, pid in enumerate(selected_for_comparison):
            prod_details = get_product_details(products, pid)
            col_idx = idx % 3
            
            with cols[col_idx]:
                display_product_image(prod_details['image_url'], width=120)
                st.markdown(f"**{prod_details['name']}**")
                st.caption(f"${prod_details['price']:.2f} | {prod_details['avg_rating']:.1f}⭐")
        
        st.divider()
        
        # Perform comparison
        with st.spinner(f"Analyzing {len(selected_for_comparison)} products with {n_workers} workers..."):
            comparison_results, elapsed = time_function(
                comparator.compare_products,
                selected_for_comparison,
                n_workers=n_workers
            )
        
        st.success(f"✓ Compared {len(comparison_results)} products in {format_time(elapsed)}")
        
        # Build comparison table
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
        st.subheader("📊 Visual Comparison")
        
        chart_data = pd.DataFrame({
            'Product': [p['name'][:20] + '...' if len(p['name']) > 20 else p['name'] 
                       for p in comparison_results],
            'Price ($)': [p['price'] for p in comparison_results],
            'Rating': [p['avg_rating'] for p in comparison_results]
        })
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.bar_chart(chart_data.set_index('Product')['Price ($)'])
        
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
          "Optimized for 50K+ products with search-based selection and image display")
