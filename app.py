import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sparks Edu: Executive Strategy Deck",
    page_icon="📊",
    layout="wide"
)

# --- 2. SIDEBAR LEGEND ---
with st.sidebar:
    st.header("Executive Summary")
    st.info("📊 **Analysis Context**\n\nComparing **efficiency** across spend levels to determine optimal scaling strategy.")
    st.markdown("---")
    st.markdown("**Definitions:**")
    st.success("🟢 **Efficient Zone**\nSpend < 2B/week")
    st.error("🔴 **Inefficient Zone**\nSpend > 2B/week")

# --- 3. LOAD DATA DYNAMICALLY ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ads_data.csv')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # FIX: HANDLE DATES & NUMBERS
        YEAR_SUFFIX = " 2019" 
        df['week_start_date'] = df['week_start_date'].astype(str) + YEAR_SUFFIX
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='%d %b %Y', errors='coerce')
        df = df.dropna(subset=['week_start_date']).sort_values('week_start_date')

        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'ads_unique_cta', 'cpa', 'cpc', 'cpm']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # METRICS
        df['unique_cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_unique_cta'] if x['ads_unique_cta'] > 0 else 0, axis=1)
        df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
        df['booking_rate'] = df.apply(lambda x: (x['ads_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
        df['calculated_cpc'] = df.apply(lambda x: x['ads_spend'] / x['ads_click'] if x['ads_click'] > 0 else 0, axis=1)
        
        # STATUS DEFINITION
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

# --- 4. MAIN DASHBOARD LAYOUT ---

if not df.empty:
    st.title("🚀 Sparks Edu: Strategic Performance Review")
    st.markdown("### FY2019 Ad Performance & Optimization Plan")
    st.divider()

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["1. Performance Snapshot", "2. Efficiency Audit", "3. Deep Dive Diagnostics", "4. Strategic Comparison"])

    # === TAB 1: SNAPSHOT ===
    with tab1:
        st.subheader("Executive Overview: Spend vs. Acquisition")
        
        # High Level Metrics
        total_spend = df['ads_spend'].sum()
        total_unique_leads = df['ads_unique_cta'].sum()
        avg_unique_cpa = total_spend / total_unique_leads if total_unique_leads > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Investment", f"Rp {total_spend/1e9:.2f} B")
        k2.metric("Total Unique Leads", f"{total_unique_leads:,.0f}")
        k3.metric("Avg Cost Per Lead", f"Rp {avg_unique_cpa:,.0f}")
        
        st.markdown("---")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(x=df['week_start_date'], y=df['ads_spend'], name='Ad Spend (Rp)', marker_color='#94a3b8', opacity=0.6))
        fig_dual.add_trace(go.Scatter(x=df['week_start_date'], y=df['ads_unique_cta'], name='Unique Leads', yaxis='y2', line=dict(color='#0f172a', width=3)))
        fig_dual.update_layout(title="Weekly Volume: Spend vs Unique Leads", yaxis=dict(title='Spend (IDR)', showgrid=False), yaxis2=dict(title='Unique Leads', overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", y=1.1), hovermode='x unified', template="plotly_white", height=450)
        st.plotly_chart(fig_dual, use_container_width=True)

    # === TAB 2: EXTREMES ===
    with tab2:
        st.subheader("Efficiency Audit: Identifying Value Leaks")
        
        fig_status = px.scatter(df, x='week_start_date', y='unique_cpa', color='status', size='ads_spend', color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#cbd5e1'}, title="Cost Efficiency Timeline (Green = Efficient, Red = Expensive)", labels={'unique_cpa': 'CPA (Rp)', 'week_start_date': 'Week'})
        st.plotly_chart(fig_status, use_container_width=True)
        
        col_good, col_bad = st.columns(2)
        top_5_good = df.nsmallest(5, 'unique_cpa')
        top_5_bad = df.nlargest(5, 'unique_cpa')
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ Top Performers (Benchmark)")
            st.dataframe(top_5_good[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}), hide_index=True, use_container_width=True)

        with col_bad:
            st.error("⚠️ Value Leaks (Inefficient)")
            st.dataframe(top_5_bad[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}), hide_index=True, use_container_width=True)

    # === TAB 3: DEEP DIVE ===
    with tab3:
        st.subheader("Diagnostic: Why does efficiency break at scale?")
        
        # --- DATA PREP ---
        low_spend = df[df['ads_spend'] < 2e9]
        high_spend = df[df['ads_spend'] >= 2e9]
        
        avg_cpc_low = low_spend['calculated_cpc'].mean()
        avg_cpc_high = high_spend['calculated_cpc'].mean()
        cpc_growth = ((avg_cpc_high - avg_cpc_low) / avg_cpc_low) * 100
        
        avg_br_low = low_spend['booking_rate'].mean()
        avg_br_high = high_spend['booking_rate'].mean()
        br_growth = ((avg_br_high - avg_br_low) / avg_br_low) * 100
        
        col_insight_1, col_insight_2 = st.columns([1, 1])
        
        with col_insight_1:
            st.markdown("#### The Unit Economics Gap")
            
            # Simple Text Comparison
            st.info(f"**When we scale (Spend > 2B):**")
            st.write(f"1. Cost Per Click jumps by **{cpc_growth:.0f}%** 🔴")
            st.write(f"2. Booking Rate improves by **{br_growth:.0f}%** 🟢")
            st.error(f"**Result:** We pay {cpc_growth:.0f}% more to get a {br_growth:.0f}% benefit. This is why we lose money.")

        with col_insight_2:
            st.markdown("#### The Visual Proof (Trend)")
            # Standard Line Chart of CPC vs Booking Rate to show the gap
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df['week_start_date'], y=df['calculated_cpc'], name='CPC (Cost)', line=dict(color='#ef4444', width=3)))
            fig_trend.add_trace(go.Scatter(x=df['week_start_date'], y=df['booking_rate'], name='Booking Rate (Quality)', yaxis='y2', line=dict(color='#22c55e', width=3)))
            
            fig_trend.update_layout(
                title="Trend: Cost skyrocketing vs Quality flatlining",
                yaxis=dict(title='CPC (IDR)', showgrid=False),
                yaxis2=dict(title='Booking Rate (%)', overlaying='y', side='right', showgrid=False),
                hovermode='x unified',
                template="plotly_white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # === TAB 4: THE STRATEGIC COMPARISON TABLE ===
    with tab4:
        st.subheader("Strategic Impact: The 2 Billion Wall")
        st.markdown("Detailed breakdown of how performance degrades when crossing the spend threshold.")
        st.divider()

        # --- DATA CALCULATION FOR COMPARISON ---
        low_spend_weeks = df[df['ads_spend'] < 2e9]
        high_spend_weeks = df[df['ads_spend'] >= 2e9]

        # Key Metrics Averages
        cpa_low = low_spend_weeks['unique_cpa'].mean()
        cpa_high = high_spend_weeks['unique_cpa'].mean()
        
        cpc_low = low_spend_weeks['calculated_cpc'].mean()
        cpc_high = high_spend_weeks['calculated_cpc'].mean()
        
        br_low = low_spend_weeks['booking_rate'].mean()
        br_high = high_spend_weeks['booking_rate'].mean()

        # Calculate % Increase
        cpa_increase = ((cpa_high - cpa_low) / cpa_low) * 100
        cpc_increase = ((cpc_high - cpc_low) / cpc_low) * 100
        br_increase = ((br_high - br_low) / br_low) * 100

        # --- VISUAL 1: THE "WALL" CHART ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### 1. Visual Comparison")
            
            fig_wall = go.Figure()
            fig_wall.add_trace(go.Bar(
                name='Efficient Zone (<2B)',
                x=['Avg Cost Per Student', 'Avg Cost Per Click'],
                y=[cpa_low, cpc_low],
                marker_color='#22c55e',
                text=[f"Rp {cpa_low:,.0f}", f"Rp {cpc_low:,.0f}"],
                textposition='auto'
            ))
            fig_wall.add_trace(go.Bar(
                name='Inefficient Zone (>2B)',
                x=['Avg Cost Per Student', 'Avg Cost Per Click'],
                y=[cpa_high, cpc_high],
                marker_color='#ef4444',
                text=[f"Rp {cpa_high:,.0f}", f"Rp {cpc_high:,.0f}"],
                textposition='auto'
            ))
            fig_wall.update_layout(title="The Cost Penalty", barmode='group', template="plotly_white", height=400)
            st.plotly_chart(fig_wall, use_container_width=True)

        with c2:
            st.markdown("#### 2. Unit Economics Impact Table")
            
            # --- THE TABLE ---
            summary_data = {
                "Metric": ["Avg Cost Per Student (CPA)", "Avg Cost Per Click (CPC)", "Avg Booking Rate (Quality)"],
                "Efficient Zone (<2B)": [f"Rp {cpa_low:,.0f}", f"Rp {cpc_low:,.0f}", f"{br_low:.2f}%"],
                "Inefficient Zone (>2B)": [f"Rp {cpa_high:,.0f}", f"Rp {cpc_high:,.0f}", f"{br_high:.2f}%"],
                "Variance (Impact)": [f"🔴 +{cpa_increase:.0f}% Cost", f"🔴 +{cpc_increase:.0f}% Cost", f"🟢 +{br_increase:.0f}% Quality"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.info("""
            **Executive Takeaway:**
            When we scale past 2B, we gain slightly better quality users (+19%), but we pay **double the price** for the clicks (+86%).
            
            This mismatch drives our Cost Per Student up by **6%**, reducing overall profitability.
            """)

        st.divider()

        # --- STRATEGIC IMPERATIVES ---
        st.subheader("Final Recommendations")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 1. Capital Discipline")
            st.markdown("Implement strict **Rp 2B Weekly Cap**. Scaling beyond this point consistently triggers the cost penalty.")
        with col_b:
            st.markdown("#### 2. Creative Engine")
            st.markdown(f"Our CPC is rising by **{cpc_increase:.0f}%**. We must launch 4 new creative angles/month to lower base costs.")
        with col_c:
            st.markdown("#### 3. Truth Metrics")
            st.markdown("Stop reporting 'Total Leads'. Use **Unique CPA** to see the true cost of growth.")

        st.caption("Generated for Sparks Edu Case Study | Data Source: 2019 Ads Export")

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
