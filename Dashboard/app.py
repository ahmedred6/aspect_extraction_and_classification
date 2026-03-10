# dashboard/app.py

import streamlit as st
import plotly.express as px
import pandas as pd

print("APP.PY LOADED SUCCESSFULLY")
# -------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Hybrid ABSA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- SIMPLE GLOBAL STYLES ------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        padding-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #666666;
        padding-bottom: 1.0rem;
    }
    .section-card {
        background-color: #ffffff;
        padding: 1.2rem 1.4rem;
        border-radius: 0.9rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR --------------------
st.sidebar.title("🔍 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📝 Review Analyzer", "📊 Dataset Overview"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Hybrid Dialect-Adaptive ABSA System")

# -------------- PAGE 1: OVERVIEW ------------

if page == "🏠 Overview":
    st.markdown('<div class="main-title">Hybrid ABSA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Visualization for the English + Arabic + Gulf Dialect Aspect-Based Sentiment Analysis system.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Project Summary")
        st.write(
            """
            This dashboard demonstrates a **hybrid ABSA pipeline**:

            - Detects language / dialect (English, MSA, Gulf dialects)  
            - Normalizes Arabic dialect → MSA using an LLM  
            - Extracts aspect terms from reviews  
            - Classifies sentiment for each aspect using classical ML ensembles  
            - Aggregates results across reviews  
            - Generates human-readable summaries with an LLM  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Architecture (High Level)")
        st.write(
            """
            - **Smart Router:** sends input to English or Arabic pipeline  
            - **Arabic branch:** dialect normalization → aspect extraction → TF-IDF → ensemble  
            - **English branch:** spaCy preprocessing → TF-IDF → ensemble  
            - **Aggregation:** combines aspect sentiment across all reviews  
            - **Summarization:** LLM turns stats into natural-language insight  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🚥 Current Status")
        st.write(
            """
            - ✅ Normalization & aspect extraction implemented  
            - ✅ Classical ML sentiment models trained (Arabic + English)  
            - ✅ Aggregation logic implemented  
            - ⏳ Final visualization & UX (this dashboard)  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 👥 Team Roles")
        st.write(
            """
            - **Maya:** Dialect / language routing  
            - **Ahmed:** Labeling & classical ML training  
            - **You:** Input pipeline, aspect extraction, feature engineering, dashboard  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PAGE 2: REVIEW ANALYZER (stub) ---

elif page == "📝 Review Analyzer":
    st.markdown('<div class="main-title">Review Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Enter a single review and explore how the pipeline processes it.</div>',
        unsafe_allow_html=True,
    )

    # For now this is just a stub UI — we'll wire it to your models next.
    review_text = st.text_area(
        "Enter a review (English / Arabic / dialect):",
        height=140,
        placeholder="مثال: البطارية ضعيفة لكن الشحن ممتاز جدًا...",
    )

    analyze = st.button("🚀 Run Analysis")

    if analyze and review_text.strip():
        st.info("In the next step, we'll connect this button to your real pipeline.")
    elif analyze:
        st.warning("Please enter a review first.")

# -------- PAGE 3: DATASET OVERVIEW (stub) ----

elif page == "📊 Dataset Overview":
    st.markdown('<div class="main-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">High-level statistics of the processed reviews (coming next).</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("In the next step we will:")
    st.write(
        """
        - Load your `extractor_output.jsonl`  
        - Show language distribution (English vs Arabic)  
        - Show top aspects and their frequency  
        - Plot sentiment distribution  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
