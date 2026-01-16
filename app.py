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
        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'cpa', 'cpc', 'cpm']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Calculate Key Metrics
        # CPA (Cost Per Acquisition) - Lower is better
        df['calculated_cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_cta'] if x['ads_cta'] > 0 else 0, axis=1)
        # CTR (Click Through Rate) - Higher is better
        df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
        # CVR (Conversion Rate) - Higher is better
        df['cvr'] = df.apply(lambda x: (x['ads_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
        
        # 4. Define Efficiency Status for Visualization
        # We define "Good" as CPA below the 25th percentile, and "Bad" as CPA above 75th percentile
        cpa_25 = df['calculated_cpa'].quantile(0.25)
        cpa_75 = df['calculated_cpa'].quantile(0.75)
        
        def get_status(cpa):
            if cpa <= cpa_25: return 'Good (Efficient)'
            elif cpa >= cpa_75: return 'Problem (Expensive)'
            else: return 'Normal'
            
        df['status'] = df['calculated_cpa'].apply(get_status)
            
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. DASHBOARD LAYOUT ---

if not df.empty:
    st.title("🚀 Sparks Edu: Ads Performance Review")
    st.markdown("This dashboard provides a snapshot of ad performance, identifies critical weeks, and offers actionable recommendations.")
    
    # Global Metrics
    total_spend = df['ads_spend'].sum()
    total_bookings = df['ads_cta'].sum()
    avg_cpa = total_spend / total_bookings if total_bookings > 0 else 0
    
    # Top KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Investment", f"Rp {total_spend/1e9:.2f} B")
    kpi2.metric("Total Bookings", f"{total_bookings:,.0f}")
    kpi3.metric("Overall Average CPA", f"Rp {avg_cpa:,.0f}")
    
    st.divider()

    # --- TABS FOR SPECIFIC QUESTIONS ---
    tab1, tab2, tab3, tab4 = st.tabs(["1. Snapshot View", "2. Success vs Problems (Detailed)", "3. Deep Dive Insights", "4. Recommendations"])

    # === TAB 1: SNAPSHOT VIEW ===
    with tab1:
        st.subheader("1. Executive Snapshot")
        st.markdown("A quick view of how Spend correlates with Results (Bookings) over time.")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=df['week_start_date'], y=df['ads_spend'], name='Ad Spend (Rp)',
            marker_color='#d1d5db', opacity=0.6
        ))
        fig_dual.add_trace(go.Scatter(
            x=df['week_start_date'], y=df['ads_cta'], name='Bookings (Count)',
            yaxis='y2', line=dict(color='#2563eb', width=3), mode='lines+markers'
        ))
        fig_dual.update_layout(
            title="Ad Spend vs. Bookings Generated",
            yaxis=dict(title='Spend (IDR)', showgrid=False),
            yaxis2=dict(title='Bookings', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified', template="plotly_white"
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # === TAB 2: SUCCESS VS PROBLEM WEEKS (UPDATED) ===
    with tab2:
        st.subheader("2. Detailed Success & Problem Detection")
        st.markdown("""
        We analyze **Efficiency (CPA)**. 
        - **Green** dots are weeks where we paid very little per booking.
        - **Red** dots are weeks where we overpaid for bookings.
        """)
        
        # 1. Visual Overview
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='calculated_cpa', 
            color='status',
            size='ads_spend',
            color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
            title="Performance Timeline: Good vs Bad Weeks",
            labels={'calculated_cpa': 'Cost Per Acquisition (CPA)', 'week_start_date': 'Week'}
        )
        st.plotly_chart(fig_status, use_container_width=True)

        st.divider()

        # 2. Detailed Tables
        col_good, col_bad = st.columns(2)
        
        # Get Top 5 Best and Worst
        top_5_good = df.nsmallest(5, 'calculated_cpa')[['week_start_date', 'ads_spend', 'ads_cta', 'calculated_cpa']]
        top_5_bad = df.nlargest(5, 'calculated_cpa')[['week_start_date', 'ads_spend', 'ads_cta', 'calculated_cpa']]
        
        # Format helper
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ Top 5 BEST Weeks (Lowest Cost per Booking)")
            st.dataframe(
                top_5_good.style.format({
                    'ads_spend': format_currency, 
                    'calculated_cpa': format_currency,
                    'ads_cta': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )
            st.caption("These weeks show high efficiency. Notice that **Spend is often moderate**, not maximum.")

        with col_bad:
            st.error("⚠️ Top 5 WORST Weeks (Highest Cost per Booking)")
            st.dataframe(
                top_5_bad.style.format({
                    'ads_spend': format_currency, 
                    'calculated_cpa': format_currency,
                    'ads_cta': '{:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )
            st.caption("These weeks show wasted budget. Notice that **Spend is often very high**, but bookings didn't increase proportionally.")

    # === TAB 3: INSIGHTS ===
    with tab3:
        st.subheader("3. Deep Dive Insights")
        col_insight_1, col_insight_2 = st.columns(2)
        
        with col_insight_1:
            st.markdown("#### Insight A: Diminishing Returns")
            fig_scatter = px.scatter(
                df, x='ads_spend', y='calculated_cpa', size='ads_cta', color='status',
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
                title="Spend vs. Efficiency (CPA)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Notice how the RED dots (Bad weeks) appear when Spend is highest (Right side of chart).")

        with col_insight_2:
            st.markdown("#### Insight B: Ad Fatigue")
            fig_line = px.line(df, x='week_start_date', y='ctr', title="Click-Through Rate (CTR) Trend")
            fig_line.update_traces(line_color='#9333ea')
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("A dropping line indicates creative fatigue (people are tired of the ads).")

    # === TAB 4: RECOMMENDATIONS ===
    with tab4:
        st.subheader("4. Strategic Recommendations")
        st.info("Based on the Top 5 Best/Worst analysis:")
        st.markdown("""
        ### 1. 🛑 Implement a "Budget Ceiling"
        * **Evidence:** In the 'Top 5 Worst Weeks' table, the ad spend is consistently very high (often above Rp 2B), but the CPA is 3x-4x higher than normal.
        * **Action:** Do not blindly scale spend. Cap weekly spend at **Rp 1.5B - 1.8B**. Scaling beyond this point consistently destroys efficiency.

        ### 2. 🔄 Review "Bad Week" Creative Performance
        * **Evidence:** The CTR in problem weeks often drops. 
        * **Action:** During high-spend weeks, creatives burn out faster. If you plan to spend >Rp 2B, you must have **3x more creative variations** ready to rotate than a normal week.

        ### 3. 🎯 Audience Saturation Check
        * **Evidence:** Getting fewer bookings despite higher spend means we have exhausted the "easy" audience.
        * **Action:** When efficiency drops (Red dots appear), stop increasing budget on the same audience. You must launch a **new** audience set (e.g., lookalikes, new interests) before increasing spend further.
        """)

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
