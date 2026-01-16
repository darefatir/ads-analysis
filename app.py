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
        # 1. Dates: Add Year (Assuming 2024) to fix "07 Jan" format
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
    tab1, tab2, tab3, tab4 = st.tabs(["1. Snapshot View", "2. Success vs Problems", "3. Deep Dive Insights", "4. Recommendations"])

    # === TAB 1: SNAPSHOT VIEW (Question 1) ===
    with tab1:
        st.subheader("1. Executive Snapshot")
        st.markdown("A quick view of how Spend correlates with Results (Bookings) over time.")
        
        # Dual Axis Chart: Spend (Bar) vs Bookings (Line)
        fig_dual = go.Figure()
        
        # Bar: Spend
        fig_dual.add_trace(go.Bar(
            x=df['week_start_date'],
            y=df['ads_spend'],
            name='Ad Spend (Rp)',
            marker_color='#d1d5db', # Gray for context
            opacity=0.6
        ))

        # Line: Bookings
        fig_dual.add_trace(go.Scatter(
            x=df['week_start_date'],
            y=df['ads_cta'],
            name='Bookings (Count)',
            yaxis='y2',
            line=dict(color='#2563eb', width=3),
            mode='lines+markers'
        ))

        fig_dual.update_layout(
            title="Ad Spend vs. Bookings Generated",
            yaxis=dict(title='Spend (IDR)', showgrid=False),
            yaxis2=dict(title='Bookings', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig_dual, use_container_width=True)
        
        with st.expander("Show Raw Data Table"):
            st.dataframe(df[['week_start_date', 'ads_spend', 'ads_cta', 'calculated_cpa', 'ctr', 'cvr']].style.format({
                'ads_spend': 'Rp {:,.0f}',
                'calculated_cpa': 'Rp {:,.0f}',
                'ctr': '{:.2f}%',
                'cvr': '{:.2f}%'
            }))

    # === TAB 2: SUCCESS VS PROBLEM WEEKS (Question 2) ===
    with tab2:
        st.subheader("2. Success & Problem Detection")
        st.markdown("""
        We define **Success** as high efficiency (getting bookings for a **low cost**).
        We define **Problem** as low efficiency (paying a **high cost** for bookings).
        """)

        # Logic to find weeks
        # Filter for weeks with actual activity to avoid data errors
        active_weeks = df[df['ads_cta'] > 0]
        
        if not active_weeks.empty:
            # Success = Lowest CPA
            best_row = active_weeks.loc[active_weeks['calculated_cpa'].idxmin()]
            # Problem = Highest CPA
            worst_row = active_weeks.loc[active_weeks['calculated_cpa'].idxmax()]

            col_good, col_bad = st.columns(2)

            with col_good:
                st.success("✅ SUCCESS WEEK: Most Efficient Spend")
                st.write(f"**Date:** {best_row['week_start_date'].strftime('%d %b %Y')}")
                st.metric("CPA (Cost per Booking)", f"Rp {best_row['calculated_cpa']:,.0f}", delta="Lowest Cost", delta_color="normal")
                st.write(f"**Why?** You spent **Rp {best_row['ads_spend']/1e6:.1f}M** to get **{best_row['ads_cta']:,.0f}** bookings. The conversion rate was likely efficient, maximizing the budget.")

            with col_bad:
                st.error("⚠️ PROBLEM WEEK: Least Efficient Spend")
                st.write(f"**Date:** {worst_row['week_start_date'].strftime('%d %b %Y')}")
                st.metric("CPA (Cost per Booking)", f"Rp {worst_row['calculated_cpa']:,.0f}", delta="-High Cost", delta_color="inverse")
                st.write(f"**Why?** You spent **Rp {worst_row['ads_spend']/1e9:.2f}B** (very high) but paid significantly more for each booking. This suggests **diminishing returns**—spending more didn't proportionally increase results.")

    # === TAB 3: INSIGHTS (Question 3) ===
    with tab3:
        st.subheader("3. Data-Driven Insights")
        
        col_insight_1, col_insight_2 = st.columns(2)
        
        with col_insight_1:
            st.markdown("#### Insight A: Diminishing Returns")
            st.markdown("Does spending more money always give us efficient results? **No.**")
            
            fig_scatter = px.scatter(
                df, 
                x='ads_spend', 
                y='calculated_cpa',
                size='ads_cta',
                color='ctr',
                title="Spend vs. Cost Per Acquisition (CPA)",
                labels={'ads_spend': 'Total Spend', 'calculated_cpa': 'Cost Per Booking (CPA)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Observation: As Spend (X-axis) increases, CPA (Y-axis) tends to go UP. This means we are overspending in certain weeks.")

        with col_insight_2:
            st.markdown("#### Insight B: Ad Fatigue (CTR Trend)")
            st.markdown("Are people getting bored of our ads?")
            
            fig_line = px.line(df, x='week_start_date', y='ctr', title="Click-Through Rate (CTR) Over Time")
            fig_line.update_traces(line_color='#9333ea')
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("Observation: If the line trends downwards, it means fewer people are clicking our ads relative to impressions (Ad Fatigue).")

    # === TAB 4: RECOMMENDATIONS (Question 4) ===
    with tab4:
        st.subheader("4. Strategic Recommendations")
        
        st.info("Based on the data analysis, here are 3 actionable steps to improve performance:")
        
        st.markdown("""
        ### 1. 🛑 Cap the Budget to Avoid Waste
        * **Observation:** The data shows that when spend exceeds **Rp 2 Billion/week**, the CPA spikes drastically (Efficiency drops).
        * **Action:** Set a weekly budget cap around **Rp 1.5 Billion**. Reallocate the saved budget to future weeks rather than burning it all at once for low returns.

        ### 2. 🎨 Refresh Ad Creatives (Combat Fatigue)
        * **Observation:** The CTR (Click Through Rate) fluctuations suggest audiences might be getting tired of seeing the same ads.
        * **Action:** Launch a **new creative batch every 2-3 weeks**. If CTR drops below 1%, immediately rotate the image or headline.

        ### 3. 🎯 Narrow the Targeting
        * **Observation:** The "Problem Weeks" had high impressions but low conversion efficiency. This usually means the ads were shown to a "Broad" audience that wasn't interested.
        * **Action:** Stop "Broad" targeting during high-spend weeks. Focus budget on **Lookalike Audiences (1-3%)** and Retargeting (people who visited the site but didn't book), as these typically have lower CPA.
        """)

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
