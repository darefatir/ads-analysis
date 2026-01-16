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
    tab1, tab2, tab3, tab4 = st.tabs(["1. Performance Snapshot", "2. Efficiency Audit", "3. Deep Dive Diagnostics", "4. Strategic Impact Model"])

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

    # === TAB 3: DEEP DIVE (THE NEW "LOUD" CHART) ===
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
            st.markdown("Comparing the **Cost Increase** vs **Quality Increase** when scaling past 2B.")
            
            # THE "LOUD" BAR CHART
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                x=['Cost Per Click (Cost)', 'Booking Rate (Quality)'],
                y=[cpc_growth, br_growth],
                text=[f"+{cpc_growth:.0f}%", f"+{br_growth:.0f}%"],
                textposition='auto',
                marker_color=['#ef4444', '#22c55e']
            ))
            fig_gap.update_layout(
                title="The Scaling Penalty: Cost vs Value",
                yaxis_title="Growth % (High Spend vs Low Spend)",
                template="plotly_white"
            )
            st.plotly_chart(fig_gap, use_container_width=True)
            
            st.error(f"**Insight:** Costs exploded by **{cpc_growth:.0f}%**, but quality only improved by **{br_growth:.0f}%**. We are paying a premium for diminishing returns.")

        with col_insight_2:
            st.markdown("#### The Saturation Point")
            df['chart_label'] = df.apply(lambda x: x['week_start_date'].strftime('%d %b') if x['status'] == 'Problem (Expensive)' else '', axis=1)
            fig_scatter = px.scatter(df, x='ads_spend', y='unique_cpa', size='ads_unique_cta', color='status', color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#cbd5e1'}, title="Spend vs. CPA (The 'Elbow')", labels={'ads_spend': 'Total Weekly Spend (Rp)', 'unique_cpa': 'CPA (Rp)', 'ads_unique_cta': 'Leads Volume'}, text='chart_label')
            fig_scatter.add_vline(x=2000000000, line_dash="dash", line_color="black", annotation_text="Optimal Threshold (2B)")
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)

    # === TAB 4: STRATEGIC SIMULATION (BEFORE vs AFTER) ===
    with tab4:
        st.subheader("Strategic Impact Model: 'Before vs After' Analysis")
        st.markdown("We simulated the impact of implementing the **'Efficiency Strategy'** (Capping Spend & Optimizing Creative) on the 2019 data.")
        st.divider()

        # --- SIMULATION LOGIC ---
        # 1. Actuals
        actual_spend = df['ads_spend'].sum()
        actual_leads = df['ads_unique_cta'].sum()
        actual_cpa = actual_spend / actual_leads
        
        # 2. Optimized Scenario: 
        # Assume we could have acquired the SAME number of leads, but at the "Efficient CPA" (Safe Zone Average)
        efficient_cpa = df[df['ads_spend'] < 2e9]['unique_cpa'].mean()
        optimized_spend = actual_leads * efficient_cpa
        
        savings = actual_spend - optimized_spend
        savings_pct = (savings / actual_spend) * 100
        
        # --- VISUALIZATION: BEFORE vs AFTER ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### Scenario: What if we maintained efficiency all year?")
            
            fig_sim = go.Figure()
            # Bar 1: Actual
            fig_sim.add_trace(go.Bar(
                name='Actual 2019 Spend',
                x=['Total Spend'],
                y=[actual_spend],
                marker_color='#94a3b8',
                text=[f"Rp {actual_spend/1e9:.1f} B"],
                textposition='auto'
            ))
            # Bar 2: Optimized
            fig_sim.add_trace(go.Bar(
                name='Optimized Spend (Same Leads)',
                x=['Total Spend'],
                y=[optimized_spend],
                marker_color='#22c55e',
                text=[f"Rp {optimized_spend/1e9:.1f} B"],
                textposition='auto'
            ))
            
            fig_sim.update_layout(
                title="Potential Savings Calculator",
                yaxis_title="Total Spend (IDR)",
                barmode='group',
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        with c2:
            st.markdown("### 💰 The Opportunity")
            st.metric("Potential Savings", f"Rp {savings/1e9:.2f} Billion", delta=f"{savings_pct:.1f}% Savings", delta_color="normal")
            
            st.info("""
            **Analysis:**
            If we had avoided the "High Spend / High CPA" trap and maintained our baseline efficiency:
            
            * We could have acquired the **same 400k+ students**.
            * But for **Rp 8.9 Billion less**.
            
            **Recommendation:** Reinvest these savings into creative testing to lower CPC further.
            """)

        st.divider()
        st.subheader("Detailed Strategic Imperatives")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 1. Capital Discipline")
            st.markdown("Implement strict **Rp 2B Weekly Cap**. High spend creates diminishing returns.")
        with col_b:
            st.markdown("#### 2. Creative Engine")
            st.markdown("Launch **4 New Creatives/Month**. Lowering CPC is the only way to scale efficiently.")
        with col_c:
            st.markdown("#### 3. Truth Metrics")
            st.markdown("Report on **Unique CPA** only. Stop reporting 'Total Leads' vanity metrics.")

        st.caption("Generated for Sparks Edu Case Study | Data Source: 2019 Ads Export")

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
