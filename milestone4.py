import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Try importing wordcloud; gracefully skip if not installed
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ---------------------- PAGE CONFIG ----------------------

st.set_page_config(
    page_title="ReviewSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# ---------------------- LOAD DATA ----------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Milestone2_Sentiment_Result_new.csv")
    except FileNotFoundError:
        st.error("❌ Could not find 'Milestone2_Sentiment_Result_new.csv'. Please make sure it is in the same folder as this script.")
        st.stop()

    # Normalize column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Rename common variations to expected names
    col_map = {}
    for c in df.columns:
        if 'sentiment' in c:
            col_map[c] = 'sentiment'
        elif 'product' in c:
            col_map[c] = 'product'
        elif 'date' in c:
            col_map[c] = 'date'
        elif 'confidence' in c:
            col_map[c] = 'confidence_score'
        elif 'review' in c or 'text' in c or 'comment' in c:
            col_map[c] = 'review_text'
    df.rename(columns=col_map, inplace=True)

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    # Ensure required columns exist with fallbacks
    if 'sentiment' not in df.columns:
        st.error("❌ No 'sentiment' column found. Columns available: " + str(list(df.columns)))
        st.stop()

    if 'product' not in df.columns:
        df['product'] = 'Unknown'

    if 'confidence_score' not in df.columns:
        df['confidence_score'] = np.nan

    # Standardize sentiment capitalization
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.capitalize()

    return df


@st.cache_data
def load_keywords():
    try:
        kdf = pd.read_csv("Milestone3_Keyword_Insights.csv")
        kdf.columns = kdf.columns.str.strip()
        return kdf
    except Exception:
        return pd.DataFrame()


df = load_data()
keywords_df = load_keywords()

# ---------------------- SIDEBAR ----------------------

st.sidebar.header("🔍 Filters")

available_sentiments = sorted(df["sentiment"].dropna().unique().tolist())
sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=available_sentiments,
    default=available_sentiments
)

available_products = sorted(df["product"].dropna().unique().tolist())
product_filter = st.sidebar.multiselect(
    "Select Product",
    options=available_products,
    default=available_products
)

st.sidebar.subheader("📅 Date Range")

valid_dates = df['date'].dropna()
if len(valid_dates) > 0:
    default_start = valid_dates.min().date()
    default_end = valid_dates.max().date()
else:
    default_start = datetime(2024, 1, 1).date()
    default_end = datetime(2025, 12, 31).date()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=default_start)
end_date = col2.date_input("End Date", value=default_end)

# ---------------------- FILTER DATA ----------------------

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# Handle NaT dates: treat rows with NaT date as always included (or excluded — here we exclude)
date_mask = (
    df['date'].isna() |
    ((df['date'] >= start_dt) & (df['date'] <= end_dt))
)

filtered_df = df[
    (df["sentiment"].isin(sentiment_filter)) &
    (df["product"].isin(product_filter)) &
    date_mask
].copy()

# ---------------------- HEADER ----------------------

st.markdown(
    '<h1 class="main-header">📊 ReviewSense – Customer Feedback Dashboard</h1>',
    unsafe_allow_html=True
)

# ---------------------- METRICS ----------------------

total_reviews = len(filtered_df)

if total_reviews == 0:
    st.warning("⚠️ No data matches the selected filters. Please adjust the sidebar filters.")
else:
    pos_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
    neg_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
    neu_count = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])

    pos_pct = pos_count / total_reviews * 100
    neg_pct = neg_count / total_reviews * 100
    neu_pct = neu_count / total_reviews * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Reviews", total_reviews)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Positive", f"{pos_pct:.1f}%", delta=f"{pos_count} reviews")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Negative", f"{neg_pct:.1f}%", delta=f"{neg_count} reviews")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Neutral", f"{neu_pct:.1f}%", delta=f"{neu_count} reviews")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- SENTIMENT DISTRIBUTION ----------------------

    st.subheader("😊 Sentiment Distribution")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    counts = filtered_df["sentiment"].value_counts()

    colors_map = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#9E9E9E'}
    bar_colors = [colors_map.get(s, '#1f77b4') for s in counts.index]
    bars = ax1.bar(counts.index, counts.values, color=bar_colors)

    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Number of Reviews")
    ax1.set_title("Overall Sentiment Breakdown")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.5,
                 int(yval), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ---------------------- PRODUCT SENTIMENT ----------------------

    st.subheader("📱 Product-wise Sentiment")

    product_sent = (
        filtered_df.groupby('product')['sentiment']
        .value_counts()
        .unstack(fill_value=0)
    )

    st.dataframe(product_sent, use_container_width=True)

    if product_sent.shape[0] > 0 and product_sent.shape[1] > 0:
        fig_hm, ax_hm = plt.subplots(figsize=(10, max(4, len(product_sent) * 0.5 + 2)))
        sns.heatmap(product_sent, annot=True, fmt="d", cmap="RdYlGn", ax=ax_hm)
        ax_hm.set_title("Product Sentiment Heatmap")
        plt.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)

    # ---------------------- TREND ----------------------

    st.subheader("📈 Sentiment Trends Over Time")

    trend_df = filtered_df.dropna(subset=['date']).copy()

    if not trend_df.empty:
        trend_df['month'] = trend_df['date'].dt.to_period('M')
        trend = trend_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

        if not trend.empty:
            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))

            line_colors = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#9E9E9E'}
            for col in trend.columns:
                ax_trend.plot(
                    trend.index.astype(str),
                    trend[col],
                    marker='o',
                    linewidth=2,
                    label=col,
                    color=line_colors.get(col, None)
                )

            ax_trend.set_xlabel("Month")
            ax_trend.set_ylabel("Number of Reviews")
            ax_trend.set_title("Monthly Sentiment Trend")
            ax_trend.legend()
            ax_trend.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_trend)
            plt.close(fig_trend)
    else:
        st.info("No valid date data available for trend analysis.")

    # ---------------------- CONFIDENCE ----------------------

    if filtered_df['confidence_score'].notna().any():
        st.subheader("📊 Confidence Score Distribution")

        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.hist(filtered_df["confidence_score"].dropna(), bins=25, color='#1f77b4', edgecolor='white')
        ax_hist.set_xlabel("Confidence Score")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Sentiment Confidence Distribution")
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

# ---------------------- KEYWORDS ----------------------

st.subheader("🔑 Top Keywords & Word Cloud")

if not keywords_df.empty:
    # Normalize keyword column names
    keywords_df.columns = keywords_df.columns.str.strip()

    # Try to find keyword and frequency columns
    kw_col = next((c for c in keywords_df.columns if 'keyword' in c.lower() or 'word' in c.lower()), None)
    freq_col = next((c for c in keywords_df.columns if 'freq' in c.lower() or 'count' in c.lower() or 'score' in c.lower()), None)

    if kw_col is None or freq_col is None:
        st.warning(f"Could not identify keyword/frequency columns. Available: {list(keywords_df.columns)}")
    else:
        keywords_df = keywords_df.rename(columns={kw_col: 'Keyword', freq_col: 'Frequency'})
        top15 = keywords_df.head(15)

        if WORDCLOUD_AVAILABLE:
            colA, colB = st.columns([3, 2])
        else:
            colA = st.container()

        with colA:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            ax_bar.barh(top15['Keyword'], top15['Frequency'], color='#1f77b4')
            ax_bar.set_xlabel("Frequency")
            ax_bar.set_title("Top Keywords")
            ax_bar.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        if WORDCLOUD_AVAILABLE:
            with colB:
                word_freq = dict(zip(keywords_df['Keyword'], keywords_df['Frequency']))
                wc = WordCloud(width=400, height=400,
                               background_color='white').generate_from_frequencies(word_freq)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 6))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout()
                st.pyplot(fig_wc)
                plt.close(fig_wc)
        else:
            st.info("💡 Install `wordcloud` (`pip install wordcloud`) to enable the Word Cloud visualization.")
else:
    st.info("No keyword data found. Make sure 'Milestone3_Keyword_Insights.csv' is present.")

# ---------------------- DATA PREVIEW & EXPORT ----------------------

with st.expander("📋 Preview Filtered Data (first 15 rows)"):
    st.dataframe(filtered_df.head(15), use_container_width=True)

st.subheader("💾 Export Options")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        "⬇️ Download Filtered Reviews",
        filtered_df.to_csv(index=False).encode('utf-8'),
        "ReviewSense_Filtered_Reviews.csv",
        "text/csv",
        use_container_width=True
    )

with col_dl2:
    if not keywords_df.empty:
        st.download_button(
            "⬇️ Download Keyword List",
            keywords_df.to_csv(index=False).encode('utf-8'),
            "ReviewSense_Keywords.csv",
            "text/csv",
            use_container_width=True
        )

st.success("✅ Dashboard ready! Use the sidebar to explore different views.")