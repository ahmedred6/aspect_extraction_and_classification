#test.py
import streamlit as st
import plotly.express as px
import pandas as pd

from load_dataset import load_all_datasets
from whole_pipeline import analyze_sentence_global

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hybrid ABSA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- GLOBAL STYLES --------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #111827 100%);
        color: #e5e7eb !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020817 !important;
        border-right: 1px solid rgba(148,163,184,0.4);
    }

    /* Cards */
    .section-card {
        background: rgba(15,23,42,0.95);
        padding: 1.1rem 1.3rem;
        border-radius: 1rem;
        box-shadow: 0 16px 40px rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.45);
        backdrop-filter: blur(16px);
        margin-bottom: 1.1rem;
    }

    .kpi-card {
        background: radial-gradient(circle at top left, #22c55e22, #0f172a);
        padding: 0.9rem 1.1rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(34,197,94,0.4);
        box-shadow: 0 14px 30px rgba(34,197,94,0.35);
    }

    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }

    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e5e7eb;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        padding-bottom: 0.2rem;
        background: linear-gradient(90deg, #a855f7, #22c55e, #38bdf8);
        -webkit-background-clip: text;
        color: transparent;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        padding-bottom: 0.8rem;
    }

    /* Plotly charts background tweak */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }

    /* Tables */
    .dataframe td, .dataframe th {
        color: #e5e7eb !important;
        background-color: #020617 !important;
        border-color: #1f2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR NAV ----------------
st.sidebar.title("🔍 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📝 Review Analyzer", "📊 Dataset Overview"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Hybrid Dialect-Adaptive ABSA System")

# ---------------- DATA LOADING (CACHED) ------

@st.cache_data
def get_data():
    df_all = load_all_datasets()
    # small cleanups
    df_all["text"] = df_all["text"].fillna("").astype(str)
    df_all["aspect"] = df_all["aspect"].fillna("").astype(str)
    df_all["language"] = df_all["language"].fillna("unknown")
    df_all["polarity"] = df_all["polarity"].fillna("unknown")
    return df_all

try:
    df = get_data()
    data_ok = True
except Exception as e:
    data_ok = False
    load_error = str(e)

# ---------------- PAGE: OVERVIEW -------------
if page == "🏠 Overview":
    st.markdown('<div class="main-title">Hybrid ABSA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Visualization for the English + Arabic + Gulf Dialect Aspect-Based Sentiment Analysis system.</div>',
        unsafe_allow_html=True,
    )

    col_top_left, col_top_right = st.columns([1.5, 1])

    with col_top_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Project Summary")
        st.write(
            """
            This dashboard showcases a **hybrid ABSA pipeline**:

            - Detects language / dialect (English, MSA, Gulf dialects)  
            - Normalizes Arabic dialect → MSA  
            - Extracts aspect terms from hotel / product reviews  
            - Classifies sentiment for each aspect via classical ML ensembles  
            - Aggregates results across reviews for both English and Arabic  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Architecture (High Level)")
        st.write(
            """
            - **Smart Router:** routes text to the English or Arabic branch  
            - **Arabic branch:** dialect routing → LLM-based normalization → ATE + ASC  
            - **English branch:** normalization → ATE + ASC  
            - **Polarity models:** overall review sentiment per language  
            - **This dashboard:** surfaces aspects, polarities, and patterns visually  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_top_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Current Status")
        st.write(
            """
            - ATE + ASC trained for English and Arabic  
            - Router model for MSA vs dialect  
            - Polarity models per language  
            - First version of interactive dashboard  
            - LLM-powered demo path integration (dialect → MSA, summaries)  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Team Roles")
        st.write(
            """
            - **Maya:** System Architecture & Dialect / language routing  
            - **Ahmed:** Pipeline integration, Labeling & classical ML training  
            - **Hamzah:** Aspect extraction, feature engineering, dashboard UX  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if data_ok:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Snapshot of Labeled Aspects")
        st.dataframe(df.sample(min(8, len(df)), random_state=42))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Dataset failed to load. Please go to **📊 Dataset Overview** for details.")

# ---------------- PAGE: REVIEW ANALYZER (still simple for now) -----
elif page == "📝 Review Analyzer":
    st.markdown("""
    <style>
        .aspects-table td {
            padding: 6px 10px;
            font-size: 15px;
        }
        .positive-pill {
            background-color: #15c47e;
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 600;
        }
        .negative-pill {
            background-color: #e63946;
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 600;
        }
        .neutral-pill {
            background-color: #457b9d;
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 600;
        }
        .section-card {
            background: rgba(255,255,255,0.08);
            padding: 18px 20px;
            border-radius: 15px;
            margin-bottom: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🔍 Review Analyzer")
    st.write("Analyze English, MSA Arabic, or Gulf dialect reviews using our **full ABSA pipeline**.")

    review_text = st.text_area(
        "Enter a review:",
        height=150,
        placeholder="Example: البطارية ممتازة لكن الكاميرا سيئة جداً..."
    )

    run_btn = st.button("🚀 Run Full Analysis")

    if run_btn:
        if not review_text.strip():
            st.warning("Please enter a review first.")
            st.stop()

        result = analyze_sentence_global(review_text)

        # -------- Routing card --------
        st.markdown("### 🌐 Language Routing")
        st.markdown(f"""
        <div class="section-card">
            <b>Detected Language:</b> {result['language'].upper()}<br>
            <b>Variant:</b> {result['variant']}<br>
            <b>Processed Sentence:</b><br>
            <span style="font-size:16px;color:#ddd;">{result['routed_sentence']}</span>
        </div>
        """, unsafe_allow_html=True)

        # -------- Overall polarity card --------
        overall = result.get("overall_polarity", "unknown")
        if str(overall).lower() == "positive":
            overall_badge = '<span class="positive-pill">Overall: Positive</span>'
        elif str(overall).lower() == "negative":
            overall_badge = '<span class="negative-pill">Overall: Negative</span>'
        else:
            overall_badge = '<span class="neutral-pill">Overall: Mixed/Neutral</span>'

        st.markdown("### 💡 Overall Sentiment")
        st.markdown(f"""
        <div class="section-card">
            {overall_badge}
        </div>
        """, unsafe_allow_html=True)

        # -------- Aspects --------
        aspects = result["absa"]["aspects"]

        st.markdown("### 🎯 Extracted Aspects")
        if not aspects:
            st.info("No aspects found in this review.")
        else:
            for asp in aspects:
                term = asp["term"]
                pol = asp["polarity"]

                if pol.lower() == "positive":
                    badge = '<span class="positive-pill">Positive</span>'
                elif pol.lower() == "negative":
                    badge = '<span class="negative-pill">Negative</span>'
                else:
                    badge = '<span class="neutral-pill">Neutral</span>'

                st.markdown(f"""
                <div class="section-card" style="border-left: 5px solid #4cc9f0;">
                    <b>Aspect:</b> {term} <br>
                    <b>Sentiment:</b> {badge} <br><br>
                    <i>Character span:</i> {asp['from']} → {asp['to']}
                </div>
                """, unsafe_allow_html=True)

        # Optional: show raw JSON
        with st.expander("🔧 Debug: raw pipeline output"):
            st.json(result)


# ---------------- PAGE: DATASET OVERVIEW -----------------
elif page == "📊 Dataset Overview":
    st.markdown('<div class="main-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">High-level statistics and visual patterns across English + Arabic aspect labels.</div>',
        unsafe_allow_html=True,
    )

    if not data_ok:
        st.error("Failed to load datasets:")
        st.code(load_error)
        st.stop()

    # ---------- KPIs ----------
    total_rows = len(df)
    total_unique_sentences = df["text"].nunique()
    total_unique_aspects = df["aspect"].nunique()

    lang_counts = df["language"].value_counts()
    polarity_counts = df["polarity"].value_counts()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])

    with c1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Total aspect labels</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{total_rows:,}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Unique sentences</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{total_unique_sentences:,}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Unique aspects</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{total_unique_aspects:,}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Languages (labels)</div>', unsafe_allow_html=True)
        lang_str = ", ".join([f"{k}: {v}" for k, v in lang_counts.items()])
        st.markdown(f'<div class="kpi-value" style="font-size:1rem;">{lang_str}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- ROW 2: Polarity + Language Distribution ----------
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🎭 Polarity Distribution")

        fig_pol = px.pie(
            df,
            names="polarity",
            hole=0.45,
            title="Aspect Polarity (All Languages)",
        )
        fig_pol.update_layout(
            legend_title_text="Polarity",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig_pol, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🌐 Language vs Polarity")

        lang_pol = df.groupby(["language", "polarity"]).size().reset_index(name="count")
        fig_lp = px.bar(
            lang_pol,
            x="language",
            y="count",
            color="polarity",
            barmode="group",
            title="Polarity by Language",
        )
        fig_lp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig_lp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- ROW 3: Top Aspects ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🏷️ Top Aspects")

top_n = st.slider("Number of top aspects to show", 5, 30, 15, 1)
col_ar, col_en = st.columns(2)

# =============================================================
# 🇸🇦 Arabic
# =============================================================
with col_ar:
    st.markdown("#### 🇸🇦 Arabic")
    df_ar = df[df["language"] == "arabic"]

    if len(df_ar) > 0:
        vc_ar = df_ar["aspect"].value_counts().head(top_n).reset_index()

        # Auto-detect names ('aspect', 'count')
        col_label, col_count = vc_ar.columns

        # Rename to stable names
        top_aspects_ar = vc_ar.rename(columns={
            col_label: "Aspect_Label",
            col_count: "Count_Value"
        })

        fig_ar = px.bar(
            top_aspects_ar,
            x="Count_Value",
            y="Aspect_Label",
            orientation="h",
            title="Most Frequent Arabic Aspects",
        )

        fig_ar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig_ar, width="stretch")
    else:
        st.info("No Arabic rows found in the dataset.")


# =============================================================
# 🇬🇧 English
# =============================================================
with col_en:
    st.markdown("#### 🇬🇧 English")
    df_en = df[df["language"] == "english"]

    if len(df_en) > 0:

        # Value counts → DataFrame
        vc_en = df_en["aspect"].value_counts().head(top_n).reset_index()

        # Auto-detect the generated column names
        col_label, col_count = vc_en.columns

        # Rename them to stable names
        top_aspects_en = vc_en.rename(columns={
            col_label: "Aspect_Label",
            col_count: "Count_Value"
        })

        fig_en = px.bar(
            top_aspects_en,
            x="Count_Value",
            y="Aspect_Label",
            orientation="h",
            title="Most Frequent English Aspects",
        )

        fig_en.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig_en, width="stretch")

    else:
        st.info("No English rows found in the dataset.")
