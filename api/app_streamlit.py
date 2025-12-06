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
    max_results: int = 60
) -> pd.DataFrame:
    """
    Search products by name (optimized for 50K+ rows).
    
    Uses vectorized string operations. Case-insensitive search on name.
    Returns up to max_results matches.
    """
    if not query or query.strip() == "":
        return pd.DataFrame()
    
    query = query.strip()
    
    # Vectorized case-insensitive search on name only
    mask = products_df['name'].str.contains(query, case=False, na=False)
    filtered = products_df[mask].head(max_results).copy()
    
    return filtered


def get_product_details(products_df: pd.DataFrame, product_id: int) -> dict:
    """
    Get detailed information for a specific product.
    """
    product_row = products_df[products_df['product_id'] == product_id]
    
    if product_row.empty:
        raise KeyError(f"Product ID {product_id} not found")
    
    row = product_row.iloc[0]
    
    return {
        'product_id': int(row['product_id']),
        'name': str(row['name']),
        'brand': str(row.get('brand', 'Unknown')),
        'category_name': str(row.get('category_name', 'Unknown')),
        'price': float(row['price']),
        'avg_rating': float(row['avg_rating']),
        'category_id': int(row['category_id']),
        'image_url': str(row.get('image_url', 'NO_IMAGE'))
    }


def display_product_image(image_url: str, width: int = 150):
    """Display product image or placeholder."""
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
    """Load product and review data (cached)."""
    data_dir = current_dir.parent / "data"
    products = load_products(str(data_dir / "products.csv"))
    reviews = load_reviews(str(data_dir / "reviews.csv"))
    validate_data(products, reviews)
    return products, reviews


@st.cache_resource
def get_models(products_df: pd.DataFrame, reviews_df: pd.DataFrame):
    """Initialize recommenders and comparator (cached)."""
    dm = DeviceManager()
    
    # Build features
    fb = FeatureBuilder(dm)
    features = fb.build_product_matrix(products_df)
    product_ids = fb.get_product_ids(products_df)
    
    # Create recommenders
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


# Load data and models
products, reviews = get_data()
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
    
    # Device info
    if dm.use_gpu:
        st.success(f"**Device:** GPU ({dm.device})")
    else:
        st.info(f"**Device:** CPU only")
    
    st.divider()
    st.subheader("📊 Dataset")
    st.metric("Products", len(products))
    st.metric("Reviews", len(reviews))


# ========== Main Content ==========
st.title("🛍️ PaReCo-Py Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Recommendations", "⚖️ Product Comparison"])


# ========== Tab 1: Recommendations ==========
with tab1:
    # Initialize session state
    if 'selected_product_id' not in st.session_state:
        st.session_state.selected_product_id = None
    if 'current_recs' not in st.session_state:
        st.session_state.current_recs = []
    if 'rec_limit' not in st.session_state:
        st.session_state.rec_limit = 6
    if 'rec_time' not in st.session_state:
        st.session_state.rec_time = 0.0
    if 'rec_mode' not in st.session_state:
        st.session_state.rec_mode = ""

    # --- View 1: Search & Grid ---
    if st.session_state.selected_product_id is None:
        st.header("Search Products")
        
        search_query = st.text_input(
            "🔍 Search for a product:",
            placeholder="e.g., television, smartphone, headphones...",
            key="main_search"
        )
        
        if search_query:
            results = search_products(products, search_query)
            
            if not results.empty:
                st.success(f"Found {len(results)} matching products")
                
                # Grid Layout for Results
                cols_per_row = 3
                rows = [results.iloc[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]
                
                for row in rows:
                    cols = st.columns(cols_per_row)
                    for idx, (_, product) in enumerate(row.iterrows()):
                        with cols[idx]:
                            # Card Container
                            with st.container(border=True):
                                display_product_image(product.get("image_url"), width=150)
                                st.markdown(f"**{product['name']}**")
                                st.caption(f"{product['avg_rating']} ⭐ | ₹{product['price']:,.2f}")
                                
                                if st.button("View Details", key=f"view_{product['product_id']}"):
                                    st.session_state.selected_product_id = int(product['product_id'])
                                    st.session_state.rec_limit = 6
                                    st.session_state.current_recs = [] # Clear old recs
                                    st.rerun()
            else:
                st.warning("No products found matching your query.")
        else:
            st.info("Start typing to search for products.")

    # --- View 2: Product Details & Recommendations ---
    else:
        pid = st.session_state.selected_product_id
        
        # Back Button
        if st.button("🔙 Back to Search"):
            st.session_state.selected_product_id = None
            st.rerun()

        try:
            details = get_product_details(products, pid)
            
            # --- Product Header ---
            st.divider()
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                display_product_image(details['image_url'], width=300)
            
            with col_info:
                st.title(details['name'])
                st.markdown(f"**Brand:** {details['brand']} | **Category:** {details['category_name']}")
                st.subheader(f"₹{details['price']:,.2f}")
                st.markdown(f"**Rating:** {details['avg_rating']} ⭐")
                
                # Auto-generated description
                desc = (
                    f"Experience the new {details['name']} by {details['brand']}. "
                    f"Top-rated in {details['category_name']} with a {details['avg_rating']} star rating. "
                    "Available now at the best price."
                )
                st.info(desc)
            
            # --- Reviews ---
            st.markdown("### 📝 Recent Reviews")
            prod_reviews = reviews[reviews['product_id'] == pid].head(20)
            
            if not prod_reviews.empty:
                with st.expander(f"View {len(prod_reviews)} Reviews", expanded=False):
                    for _, r in prod_reviews.iterrows():
                        st.markdown(f"**{r['rating']}⭐** - {r['review_text']}")
                        st.divider()
            else:
                st.caption("No reviews available for this product yet.")
            
            # --- Recommendations Logic ---
            if not st.session_state.current_recs:
                with st.spinner("Computing recommendations..."):
                    target_idx = products[products['product_id'] == pid].index[0]
                    
                    # Use GPU if available/preferred, else CPU
                    if dm.use_gpu and recommender_gpu:
                         recs, elapsed = time_function(
                            recommender_gpu.recommend_similar_gpu,
                            target_idx, top_k=20
                        )
                         mode = "GPU"
                    else:
                        recs, elapsed = time_function(
                            recommender_cpu.recommend_similar,
                            target_idx, top_k=20, n_workers=4
                        )
                        mode = "CPU"
                    
                    st.session_state.current_recs = recs
                    st.session_state.rec_time = elapsed
                    st.session_state.rec_mode = mode
            
            # --- Display Recommendations ---
            st.divider()
            st.subheader("🎯 Recommended Products")
            st.caption(f"Computed in {format_time(st.session_state.rec_time)} using {st.session_state.rec_mode}")
            
            current_recs = st.session_state.current_recs
            limit = st.session_state.rec_limit
            visible_recs = current_recs[:limit]
            
            # Grid display for recs
            r_rows = [visible_recs[i:i + 3] for i in range(0, len(visible_recs), 3)]
            
            for row in r_rows:
                r_cols = st.columns(3)
                for idx, r_item in enumerate(row):
                    r_detail = get_product_details(products, r_item['product_id'])
                    with r_cols[idx]:
                         with st.container(border=True):
                            display_product_image(r_detail['image_url'], width=100)
                            st.markdown(f"**{r_detail['name']}**")
                            st.caption(f"₹{r_detail['price']:,.2f} | {r_detail['avg_rating']}⭐")
                            st.caption(f"Similarity: {r_item['score']:.4f}")
                            
                            # Click to switch product
                            if st.button("View", key=f"rec_{r_detail['product_id']}"):
                                st.session_state.selected_product_id = int(r_detail['product_id'])
                                st.session_state.rec_limit = 6
                                st.session_state.current_recs = []
                                st.rerun()

            # --- "Recommend More" Button ---
            if limit < 20 and limit < len(current_recs):
                if st.button("⬇️ Recommend More"):
                    st.session_state.rec_limit = min(limit + 6, 20)
                    st.rerun()
            elif limit >= 20:
                st.caption("✨ Max recommendations shown.")

        except KeyError:
            st.error("Product details not found.")


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
