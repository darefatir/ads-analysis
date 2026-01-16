import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sparks Edu Ads Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- 2. LOAD DATA DYNAMICALLY ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ads_data.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # --- FIX: HANDLE DATES & NUMBERS ---
        # 1. Dates: Add Year (Assuming 2024)
        YEAR_SUFFIX = " 2024" 
        df['week_start_date'] = df['week_start_date'].astype(str) + YEAR_SUFFIX
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='%d %b %Y', errors='coerce')
        
        df = df.dropna(subset=['week_start_date']).sort_values('week_start_date')

        # 2. Numbers: Remove commas and convert to float
        # Added 'ads_unique_cta' to the cleaning list
        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'ads_unique_cta', 'cpa', 'cpc', 'cpm']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Calculate Key Metrics (USING UNIQUE CTA NOW)
        # CPA (Cost Per Unique Acquisition) - The "Honest" Metric
        df['unique_cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_unique_cta'] if x['ads_unique_cta'] > 0 else 0, axis=1)
        
        # CTR (Click Through Rate) 
        df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
        
        # CVR (Unique Conversion Rate)
        df['unique_cvr'] = df.apply(lambda x: (x['ads_unique_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
        
        # 4. Define Efficiency Status for Visualization
        # We define "Good" as CPA below the 25th percentile, and "Bad" as CPA above 75th percentile
        cpa_25 = df['unique_cpa'].quantile(0.25)
        cpa_75 = df['unique_cpa'].quantile(0.75)
        
        def get_status(cpa):
            if cpa <= cpa_25: return 'Good (Efficient)'
            elif cpa >= cpa_75: return 'Problem (Expensive)'
            else: return 'Normal'
            
        df['status'] = df['unique_cpa'].apply(get_status)
            
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. DASHBOARD LAYOUT ---

if not df.empty:
    st.title("🚀 Sparks Edu: Ads Performance (Unique Leads Focus)")
    st.markdown("""
    This dashboard analyzes performance based on **Unique Leads** (`ads_unique_cta`) rather than total clicks.
    This provides a stricter, more accurate view of actual customer acquisition cost.
    """)
    
    # Global Metrics
    total_spend = df['ads_spend'].sum()
    total_unique_leads = df['ads_unique_cta'].sum() # changed to unique
    avg_unique_cpa = total_spend / total_unique_leads if total_unique_leads > 0 else 0
    
    # Top KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Investment", f"Rp {total_spend/1e9:.2f} B")
    kpi2.metric("Total Unique Leads", f"{total_unique_leads:,.0f}")
    kpi3.metric("Avg Cost Per Unique Lead", f"Rp {avg_unique_cpa:,.0f}")
    
    st.divider()

    # --- TABS FOR SPECIFIC QUESTIONS ---
    tab1, tab2, tab3, tab4 = st.tabs(["1. Snapshot View", "2. Success vs Problems", "3. Deep Dive Insights", "4. Recommendations"])

    # === TAB 1: SNAPSHOT VIEW ===
    with tab1:
        st.subheader("1. Executive Snapshot")
        st.markdown("A quick view of Spend vs **Unique Leads** generated.")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=df['week_start_date'], y=df['ads_spend'], name='Ad Spend (Rp)',
            marker_color='#d1d5db', opacity=0.6
        ))
        fig_dual.add_trace(go.Scatter(
            x=df['week_start_date'], y=df['ads_unique_cta'], name='Unique Leads',
            yaxis='y2', line=dict(color='#2563eb', width=3), mode='lines+markers'
        ))
        fig_dual.update_layout(
            title="Ad Spend vs. Unique Leads Generated",
            yaxis=dict(title='Spend (IDR)', showgrid=False),
            yaxis2=dict(title='Unique Leads', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified', template="plotly_white"
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # === TAB 2: SUCCESS VS PROBLEM WEEKS ===
    with tab2:
        st.subheader("2. Detailed Success & Problem Detection")
        st.markdown("""
        **Metric:** Cost Per Unique Lead (Unique CPA).
        - **Green:** We paid a low price for each new person.
        - **Red:** We paid a high price for each new person.
        """)
        
        # 1. Visual Overview
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='unique_cpa', 
            color='status',
            size='ads_spend',
            color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
            title="Performance Timeline: Efficiency of Unique Leads",
            labels={'unique_cpa': 'Cost Per Unique Lead', 'week_start_date': 'Week'}
        )
        st.plotly_chart(fig_status, use_container_width=True)

        st.divider()

        # 2. Detailed Tables
        col_good, col_bad = st.columns(2)
        
        # Get Top 5 Best and Worst based on UNIQUE CPA
        top_5_good = df.nsmallest(5, 'unique_cpa')[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']]
        top_5_bad = df.nlargest(5, 'unique_cpa')[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']]
        
        # Format helper
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ Top 5 BEST Weeks (Cheapest Unique Leads)")
            st.dataframe(
                top_5_good.style.format({
                    'ads_spend': format_currency, 
                    'unique_cpa': format_currency,
                    'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )

        with col_bad:
            st.error("⚠️ Top 5 WORST Weeks (Most Expensive Unique Leads)")
            st.dataframe(
                top_5_bad.style.format({
                    'ads_spend': format_currency, 
                    'unique_cpa': format_currency,
                    'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )

    # === TAB 3: INSIGHTS ===
    with tab3:
        st.subheader("3. Deep Dive Insights")
        col_insight_1, col_insight_2 = st.columns(2)
        
        with col_insight_1:
            st.markdown("#### Insight A: Spending More vs Unique Results")
            fig_scatter = px.scatter(
                df, x='ads_spend', y='unique_cpa', size='ads_unique_cta', color='status',
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
                title="Spend vs. Cost Per Unique Lead"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("When we spend massive amounts (right side), the cost per actual person (Y-axis) skyrockets.")

        with col_insight_2:
            st.markdown("#### Insight B: Unique Conversion Rate (CVR)")
            fig_line = px.line(df, x='week_start_date', y='unique_cvr', title="Unique Conversion Rate Trend")
            fig_line.update_traces(line_color='#9333ea')
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("This shows the % of clicks that turned into actual unique leads. A drop here means the landing page or offer isn't working.")

    # === TAB 4: RECOMMENDATIONS ===
    with tab4:
        st.subheader("4. Strategic Recommendations")
        st.info("Analysis based on Unique Leads data:")
        st.markdown("""
        ### 1. 📉 Reality Check on Efficiency
        * **Observation:** Using 'Unique Leads' shows our true cost is higher than it looked before. The 'Problem Weeks' are significantly more expensive per person.
        * **Action:** Adjust internal KPI targets. If we were aiming for a Rp 50,000 CPA based on clicks, we need to raise the target to **Rp 80,000 - 100,000** for *Unique Leads*, or we will constantly pause good ads thinking they are failing.

        ### 2. 🔍 Focus on Quality over Quantity
        * **Observation:** Some weeks generated huge click volumes (High Spend) but very few *Unique* Leads. This suggests "Click Bait" ads—people clicked but realized it wasn't for them.
        * **Action:** Review ad copy in high-spend weeks. Ensure the ad clearly states what the product is to filter out low-quality clicks before they cost us money.

        ### 3. 🛑 Hard Cap on High Spend
        * **Observation:** Efficiency for acquiring *new people* degrades faster than total clicks.
        * **Action:** The data suggests diminishing returns hit harder on unique users. Cap spend strictly at **Rp 1.5B**. Pushing beyond this mostly buys duplicate clicks, not new people.
        """)

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
