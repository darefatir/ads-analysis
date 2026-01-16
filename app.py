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
# This function caches the data so it doesn't reload every time you click a button
@st.cache_data
def load_data():
    # Attempt to read the CSV file
    try:
        # Ensure 'ads_data.csv' is in the same GitHub folder as this script
        df = pd.read_csv('ads_data.csv')
        
        # Standardize Column Names (stripping spaces just in case)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert Date
        # 'dayfirst=True' helps pandas understand 7/1/2019 is Jan 7th, not July 1st
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
        
        # Calculate calculated metrics if they aren't in the CSV
        df['cpa'] = df['ads_spend'] / df['ads_cta']
        df['ctr'] = (df['ads_click'] / df['ads_impression']) * 100
        df['cvr'] = (df['ads_cta'] / df['ads_click']) * 100
        
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'ads_data.csv' not found. Please upload it to your GitHub repo.")
        return pd.DataFrame() # Return empty if failed

df = load_data()

if not df.empty:
    # --- 3. DASHBOARD HEADER ---
    st.title("📊 Sparks Edu: Ads Performance Analysis")
    
    # Date Filter (Optional: Allows user to pick the range)
    min_date = df['week_start_date'].min()
    max_date = df['week_start_date'].max()
    
    st.caption(f"Data Range: {min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}")

    # Top Level Metrics (Aggregated)
    total_spend = df['ads_spend'].sum()
    total_bookings = df['ads_cta'].sum()
    avg_cpa = total_spend / total_bookings
    avg_cvr = df['cvr'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", f"Rp {total_spend/1e9:.2f} B")
    c2.metric("Total Bookings", f"{total_bookings:,.0f}")
    c3.metric("Avg CPA", f"Rp {avg_cpa:,.0f}")
    c4.metric("Avg CVR", f"{avg_cvr:.2f}%")

    st.divider()

    # --- 4. VISUALIZATIONS ---
    
    # Question 1: Spend vs Efficiency (Dynamic)
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
        template="plotly_white"
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # Question 2: Success & Problem Detector (Dynamic)
    st.subheader("2. Automated Performance Detection")
    
    # Logic to find Best/Worst weeks dynamically based on entire dataset
    best_week = df.loc[df['cpa'].idxmin()]
    worst_week = df.loc[df['cpa'].idxmax()]

    col_success, col_problem = st.columns(2)

    with col_success:
        st.success(f"🏆 Best Week: {best_week['week_start_date'].strftime('%d %b %Y')}")
        st.write(f"**CPA:** Rp {best_week['cpa']:,.0f}")
        st.write(f"**Bookings:** {best_week['ads_cta']:,.0f}")

    with col_problem:
        st.error(f"⚠️ Worst Week: {worst_week['week_start_date'].strftime('%d %b %Y')}")
        st.write(f"**CPA:** Rp {worst_week['cpa']:,.0f}")
        st.write(f"**Bookings:** {worst_week['ads_cta']:,.0f}")

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
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
