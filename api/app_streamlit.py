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

# Initialize Session State (Global)
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
if 'compare_products' not in st.session_state:
    st.session_state.compare_products = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None


# --- Helpers ---
def add_to_compare(product_id: int):
    """Add product to comparison list."""
    if product_id not in st.session_state.compare_products:
        st.session_state.compare_products.append(product_id)
        st.toast(f"Added product {product_id} to comparison", icon="✅")
    else:
        st.toast("Product already in comparison list", icon="ℹ️")

def remove_from_compare(product_id: int):
    """Remove product from comparison list."""
    if product_id in st.session_state.compare_products:
        st.session_state.compare_products.remove(product_id)
        st.toast(f"Removed product {product_id}", icon="🗑️")
        st.rerun()


# Create tabs
tab1, tab2 = st.tabs(["🎯 Recommendations", "⚖️ Product Comparison"])


# ========== Tab 1: Recommendations ==========
with tab1:
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
                        pid = int(product['product_id'])
                        with cols[idx]:
                            with st.container(border=True):
                                display_product_image(product.get("image_url"), width=150)
                                st.markdown(f"**{product['name']}**")
                                st.caption(f"{product['avg_rating']} ⭐ | ₹{product['price']:,.2f}")
                                
                                c1, c2 = st.columns(2)
                                if c1.button("View Details", key=f"view_{pid}"):
                                    st.session_state.selected_product_id = pid
                                    st.session_state.rec_limit = 6
                                    st.session_state.current_recs = [] 
                                    st.rerun()
                                
                                if c2.button("Add Compare", key=f"comp_{pid}"):
                                    add_to_compare(pid)
            else:
                st.warning("No products found matching your query.")
        else:
            st.info("Start typing to search for products.")

    # --- View 2: Product Details & Recommendations ---
    else:
        pid = st.session_state.selected_product_id
        
        # Navigation
        c1, c2 = st.columns([1, 5])
        if c1.button("🔙 Back to Search"):
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
                
                if st.button("Add to Compare", key=f"btn_add_compare_{pid}"):
                    add_to_compare(pid)
                
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
                    r_pid = int(r_item['product_id'])
                    r_detail = get_product_details(products, r_pid)
                    with r_cols[idx]:
                         with st.container(border=True):
                            display_product_image(r_detail['image_url'], width=100)
                            st.markdown(f"**{r_detail['name']}**")
                            st.caption(f"₹{r_detail['price']:,.2f} | {r_detail['avg_rating']}⭐")
                            st.caption(f"Similarity: {r_item['score']:.4f}")
                            
                            b1, b2 = st.columns(2)
                            if b1.button("View", key=f"rec_view_{r_pid}"):
                                st.session_state.selected_product_id = r_pid
                                st.session_state.rec_limit = 6
                                st.session_state.current_recs = []
                                st.rerun()
                            
                            if b2.button("Compare", key=f"rec_comp_{r_pid}"):
                                add_to_compare(r_pid)

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
    st.header("⚖️ Product Comparison")
    
    compare_ids = st.session_state.compare_products
    
    # 1. Selected Products
    if compare_ids:
        st.subheader(f"Selected ({len(compare_ids)})")
        # Grid layout for selected products removal
        sel_rows = [compare_ids[i:i + 4] for i in range(0, len(compare_ids), 4)]
        
        for r_ids in sel_rows:
             r_cols = st.columns(4)
             for idx, pid in enumerate(r_ids):
                with r_cols[idx]:
                    try:
                        p_info = get_product_details(products, pid)
                        with st.container(border=True):
                            st.caption(f"**{p_info['name']}**")
                            # display_product_image(p_info['image_url'], width=80) 
                            if st.button("❌ Remove", key=f"rem_{pid}"):
                                remove_from_compare(pid)
                    except KeyError:
                        st.error(f"ID {pid} not found")
        st.divider()
    else:
        st.info("No products selected. Add products from the Search tab search below.")

    # 2. Add More Products
    st.subheader("Add Products to Compare")
    comp_search = st.text_input("Search (e.g. monitor, laptop) to add:", key="comp_search_bar")
    
    if comp_search:
        results = search_products(products, comp_search, max_results=12)
        if not results.empty:
            rows = [results.iloc[i:i + 4] for i in range(0, len(results), 4)]
            for row in rows:
                cols = st.columns(4)
                for idx, (_, p) in enumerate(row.iterrows()):
                    pid = int(p['product_id'])
                    with cols[idx]:
                        with st.container(border=True):
                            st.caption(f"**{p['name']}**")
                            st.caption(f"₹{p['price']:,.0f}")
                            
                            if pid in compare_ids:
                                st.button("✓ Added", disabled=True, key=f"added_{pid}")
                            else:
                                if st.button("Add", key=f"add_comp_search_{pid}"):
                                    add_to_compare(pid)
                                    st.rerun()

    # 3. Compare Action
    st.divider()
    
    if len(compare_ids) >= 2:
        if st.button("🚀 Compare Now", type="primary"):
            st.subheader("Comparison Results")
            
            comp_data = []
            for pid in compare_ids:
                d = get_product_details(products, pid)
                comp_data.append(d)
                
            df_comp = pd.DataFrame(comp_data)
            
            # Metric Calculation
            # Weighted Score: Rating (normalized) * 0.7 + (1/Price normalized) * 0.3 approx
            # Simple heuristic: Rating * 10 - Price/1000
            df_comp['score'] = (df_comp['avg_rating'] * 2000) - df_comp['price']
            
            best_overall = df_comp.loc[df_comp['score'].idxmax()]
            best_perf = df_comp.loc[df_comp['avg_rating'].idxmax()]
            
            # Budget: min price among those with rating >= 4.0, else min price
            budg_cand = df_comp[df_comp['avg_rating'] >= 4.0]
            if budg_cand.empty:
                 budg_cand = df_comp
            best_budget = budg_cand.loc[budg_cand['price'].idxmin()]

            # Badges
            c1, c2, c3 = st.columns(3)
            c1.success(f"🏆 Best Overall: **{best_overall['name']}**")
            c2.info(f"💰 Best Budget: **{best_budget['name']}**")
            c3.warning(f"⭐ Best Performance: **{best_perf['name']}**")
            
            # Table
            st.dataframe(
                df_comp[['name', 'brand', 'price', 'avg_rating', 'category_name']].style.format({
                    'price': '₹{:.2f}',
                    'avg_rating': '{:.1f}⭐'
                }),
                use_container_width=True
            )
            
            # Workers slider hidden/fixed for simplicity or shown if needed
            
    elif len(compare_ids) == 1:
        st.warning("Select at least one more product to compare.")
    
    # 4. Footer
    st.divider()
    st.caption("Add products from the Search tab or Recommendations to build your comparison list.")
