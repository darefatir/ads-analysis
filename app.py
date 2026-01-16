import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sparks Edu Ads Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 2. LOAD DATA DYNAMICALLY ---
@st.cache_data
def load_data():
    try:
        # Read the CSV
        df = pd.read_csv('ads_data.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # --- FIX 1: CLEAN NUMERIC COLUMNS (Remove commas) ---
        # List of columns that might contain commas
        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'ads_unique_cta', 'cpm', 'cpc', 'cpa']
        
        for col in numeric_cols:
            if col in df.columns:
                # Force convert to string, remove commas, then convert to numeric
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # ----------------------------------------------------

        # --- FIX 2: DATE HANDLING ---
        # errors='coerce' turns bad data into NaT instead of crashing
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True, errors='coerce')
        
        # Drop rows where the date conversion failed (removes empty rows/totals)
        df = df.dropna(subset=['week_start_date'])
        
        # Sort by date to make charts look correct
        df = df.sort_values('week_start_date')

        # --- CALCULATE METRICS ---
        if 'ads_spend' in df.columns and 'ads_cta' in df.columns:
            # Avoid division by zero
            df['cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_cta'] if x['ads_cta'] > 0 else 0, axis=1)
        
        if 'ads_click' in df.columns and 'ads_impression' in df.columns:
            df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
            
        if 'ads_cta' in df.columns and 'ads_click' in df.columns:
            df['cvr'] = df.apply(lambda x: (x['ads_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
            
        return df

    except FileNotFoundError:
        st.error("❌ Error: 'ads_data.csv' not found. Please upload it to your GitHub repo.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- 3. DASHBOARD HEADER ---
    st.title("📊 Sparks Edu: Ads Performance Analysis")
    
    # Date Filter Info
    min_date = df['week_start_date'].min()
    max_date = df['week_start_date'].max()
    
    st.caption(f"Data Range: {min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}")

    # Top Level Metrics
    total_spend = df['ads_spend'].sum()
    total_bookings = df['ads_cta'].sum()
    
    # Safe calculation for averages
    avg_cpa = total_spend / total_bookings if total_bookings > 0 else 0
    avg_cvr = df['cvr'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", f"Rp {total_spend/1e9:.2f} B")
    c2.metric("Total Bookings", f"{total_bookings:,.0f}")
    c3.metric("Avg CPA", f"Rp {avg_cpa:,.0f}")
    c4.metric("Avg CVR", f"{avg_cvr:.2f}%")

    st.divider()

    # --- 4. VISUALIZATIONS ---
    
    # Question 1: Spend vs Efficiency
    st.subheader("1. Spend vs. Efficiency Trend")
    
    fig_combo = go.Figure()
    fig_combo.add_trace(go.Bar(
        x=df['week_start_date'], 
        y=df['ads_spend'], 
        name='Ad Spend',
        marker_color='#6366f1'
    ))
    fig_combo.add_trace(go.Scatter(
        x=df['week_start_date'], 
        y=df['cpa'], 
        name='CPA', 
        yaxis='y2',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig_combo.update_layout(
        yaxis=dict(title='Spend (IDR)', showgrid=False),
        yaxis2=dict(title='CPA (IDR)', overlaying='y', side='right', showgrid=False),
        hovermode='x unified',
        template="plotly_white",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # Question 2: Success & Problem Detector
    st.subheader("2. Automated Performance Detection")
    
    # Find Best/Worst weeks (filtering out 0 CPA to avoid empty data skews)
    valid_weeks = df[df['cpa'] > 0]
    
    if not valid_weeks.empty:
        best_week = valid_weeks.loc[valid_weeks['cpa'].idxmin()]
        worst_week = valid_weeks.loc[valid_weeks['cpa'].idxmax()]

        col_success, col_problem = st.columns(2)

        with col_success:
            st.success(f"🏆 Best Efficiency (Lowest CPA): {best_week['week_start_date'].strftime('%d %b %Y')}")
            st.metric("CPA", f"Rp {best_week['cpa']:,.0f}")
            st.metric("Bookings", f"{best_week['ads_cta']:,.0f}")

        with col_problem:
            st.error(f"⚠️ Lowest Efficiency (Highest CPA): {worst_week['week_start_date'].strftime('%d %b %Y')}")
            st.metric("CPA", f"Rp {worst_week['cpa']:,.0f}")
            st.metric("Bookings", f"{worst_week['ads_cta']:,.0f}")
    else:
        st.info("Not enough data to determine best/worst weeks.")

    # Question 3: Insights (Scatter Plot)
    st.subheader("3. Scale Analysis (Spend vs Result)")
    st.markdown("Does spending more result in more bookings? (Ideal: Dots move top-right)")
    
    fig_scatter = px.scatter(
        df, 
        x='ads_spend', 
        y='ads_cta', 
        color='cpa', 
        size='ads_spend',
        hover_data=['week_start_date'],
        labels={'ads_spend': 'Ad Spend', 'ads_cta': 'Bookings', 'cpa': 'Cost Per Acquisition'},
        title="Correlation: Spend vs Bookings"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("👋 Waiting for data. Please ensure 'ads_data.csv' is in your repository and formatted correctly.")
